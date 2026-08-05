use super::{
    CpuRefError, CpuRefErrorCode, dimension_error, invalid_image_geometry, require_finite,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultimodalRopePositions {
    pub position_ids: [Vec<i64>; 3],
    pub rope_delta: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MergedImageGrid {
    temporal: usize,
    height: usize,
    width: usize,
    tokens: usize,
}

pub fn image_placeholder_count(
    image_grid_thw: &[[usize; 3]],
    spatial_merge_size: usize,
) -> Result<usize, CpuRefError> {
    let (_, total) = validate_image_grids(image_grid_thw, spatial_merge_size)?;
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
pub fn mrope_position_ids(
    input_ids: &[u32],
    attention_mask: Option<&[u8]>,
    image_grid_thw: &[[usize; 3]],
    image_token_id: u32,
    vision_start_token_id: u32,
    spatial_merge_size: usize,
) -> Result<MultimodalRopePositions, CpuRefError> {
    if input_ids.is_empty() || image_token_id == vision_start_token_id {
        return Err(invalid_sequence_boundaries());
    }
    let (grids, _) = validate_image_grids(image_grid_thw, spatial_merge_size)?;
    let active_indices = active_token_indices(input_ids.len(), attention_mask)?;
    let active_tokens = active_indices
        .iter()
        .map(|index| input_ids[*index])
        .collect::<Vec<_>>();

    if grids.is_empty() {
        if active_tokens
            .iter()
            .any(|token| *token == image_token_id || *token == vision_start_token_id)
        {
            return Err(invalid_sequence_boundaries());
        }
        return text_only_positions(input_ids.len(), &active_indices);
    }

    let active_positions = image_positions(
        &active_tokens,
        &grids,
        image_token_id,
        vision_start_token_id,
    )?;
    let active_max = active_positions
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .ok_or_else(invalid_sequence_boundaries)?;
    let mut position_ids = std::array::from_fn(|_| vec![1_i64; input_ids.len()]);
    for (active_index, input_index) in active_indices.iter().copied().enumerate() {
        for axis in 0..3 {
            position_ids[axis][input_index] = active_positions[axis][active_index];
        }
    }
    Ok(MultimodalRopePositions {
        position_ids,
        rope_delta: checked_rope_delta(active_max, input_ids.len())?,
    })
}

pub fn decode_mrope_position_ids(
    cache_position: usize,
    sequence_length: usize,
    rope_delta: i64,
) -> Result<[Vec<i64>; 3], CpuRefError> {
    if sequence_length == 0 {
        return Err(invalid_sequence_boundaries());
    }
    let cache_position =
        i64::try_from(cache_position).map_err(|_| invalid_sequence_boundaries())?;
    let start = cache_position
        .checked_add(rope_delta)
        .filter(|position| *position >= 0)
        .ok_or_else(invalid_sequence_boundaries)?;
    let last_offset =
        i64::try_from(sequence_length - 1).map_err(|_| invalid_sequence_boundaries())?;
    start
        .checked_add(last_offset)
        .ok_or_else(invalid_sequence_boundaries)?;

    let mut positions = Vec::new();
    positions
        .try_reserve_exact(sequence_length)
        .map_err(|_| invalid_sequence_boundaries())?;
    for offset in 0..sequence_length {
        let offset = i64::try_from(offset).map_err(|_| invalid_sequence_boundaries())?;
        positions.push(
            start
                .checked_add(offset)
                .ok_or_else(invalid_sequence_boundaries)?,
        );
    }
    Ok([positions.clone(), positions.clone(), positions])
}

pub fn assemble_multimodal_embeddings_f32(
    token_embeddings: &[f32],
    projected_image_embeddings: &[f32],
    input_ids: &[u32],
    hidden_size: usize,
    image_token_id: u32,
) -> Result<Vec<f32>, CpuRefError> {
    if input_ids.is_empty() {
        return Err(invalid_sequence_boundaries());
    }
    if hidden_size == 0 {
        return Err(dimension_error());
    }
    let token_elements = input_ids
        .len()
        .checked_mul(hidden_size)
        .ok_or_else(dimension_error)?;
    if token_embeddings.len() != token_elements {
        return Err(dimension_error());
    }
    let image_rows = input_ids
        .iter()
        .filter(|token| **token == image_token_id)
        .count();
    let projected_elements = image_rows
        .checked_mul(hidden_size)
        .ok_or_else(dimension_error)?;
    if projected_image_embeddings.len() != projected_elements {
        return Err(dimension_error());
    }
    require_finite(token_embeddings)?;
    require_finite(projected_image_embeddings)?;

    let mut assembled = token_embeddings.to_vec();
    let mut projected_row = 0_usize;
    for (row, token) in input_ids.iter().copied().enumerate() {
        if token != image_token_id {
            continue;
        }
        let destination_start = row.checked_mul(hidden_size).ok_or_else(dimension_error)?;
        let source_start = projected_row
            .checked_mul(hidden_size)
            .ok_or_else(dimension_error)?;
        assembled[destination_start..destination_start + hidden_size]
            .copy_from_slice(&projected_image_embeddings[source_start..source_start + hidden_size]);
        projected_row += 1;
    }
    debug_assert_eq!(projected_row, image_rows);
    Ok(assembled)
}

fn validate_image_grids(
    image_grid_thw: &[[usize; 3]],
    spatial_merge_size: usize,
) -> Result<(Vec<MergedImageGrid>, usize), CpuRefError> {
    if spatial_merge_size == 0 || spatial_merge_size.checked_mul(spatial_merge_size).is_none() {
        return Err(invalid_image_geometry());
    }
    let mut grids = Vec::with_capacity(image_grid_thw.len());
    let mut total = 0_usize;
    for &[temporal, height, width] in image_grid_thw {
        if temporal == 0
            || height == 0
            || width == 0
            || !height.is_multiple_of(spatial_merge_size)
            || !width.is_multiple_of(spatial_merge_size)
        {
            return Err(invalid_image_geometry());
        }
        let height = height / spatial_merge_size;
        let width = width / spatial_merge_size;
        let tokens = temporal
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(invalid_image_geometry)?;
        total = total
            .checked_add(tokens)
            .ok_or_else(invalid_image_geometry)?;
        grids.push(MergedImageGrid {
            temporal,
            height,
            width,
            tokens,
        });
    }
    Ok((grids, total))
}

fn active_token_indices(
    input_length: usize,
    attention_mask: Option<&[u8]>,
) -> Result<Vec<usize>, CpuRefError> {
    let Some(mask) = attention_mask else {
        return Ok((0..input_length).collect());
    };
    if mask.len() != input_length || mask.iter().any(|value| *value > 1) {
        return Err(invalid_sequence_boundaries());
    }
    let active = mask
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == 1).then_some(index))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return Err(CpuRefError::new(
            CpuRefErrorCode::AllMasked,
            "attention mask must retain at least one token",
        ));
    }
    Ok(active)
}

fn text_only_positions(
    input_length: usize,
    active_indices: &[usize],
) -> Result<MultimodalRopePositions, CpuRefError> {
    let mut position_ids = std::array::from_fn(|_| vec![1_i64; input_length]);
    for (position, input_index) in active_indices.iter().copied().enumerate() {
        let position = i64::try_from(position).map_err(|_| invalid_sequence_boundaries())?;
        for axis in &mut position_ids {
            axis[input_index] = position;
        }
    }
    let max_position = position_ids[0]
        .iter()
        .copied()
        .max()
        .ok_or_else(invalid_sequence_boundaries)?;
    Ok(MultimodalRopePositions {
        position_ids,
        rope_delta: checked_rope_delta(max_position, input_length)?,
    })
}

fn image_positions(
    input_tokens: &[u32],
    grids: &[MergedImageGrid],
    image_token_id: u32,
    vision_start_token_id: u32,
) -> Result<[Vec<i64>; 3], CpuRefError> {
    let mut positions = std::array::from_fn(|_| Vec::with_capacity(input_tokens.len()));
    let mut cursor = 0_usize;
    let mut grid_index = 0_usize;
    let mut next_position = 0_i64;

    while let Some(relative_start) = input_tokens[cursor..]
        .iter()
        .position(|token| *token == vision_start_token_id)
    {
        let vision_start = cursor
            .checked_add(relative_start)
            .ok_or_else(invalid_sequence_boundaries)?;
        if input_tokens[cursor..vision_start].contains(&image_token_id) {
            return Err(invalid_sequence_boundaries());
        }
        let grid = grids
            .get(grid_index)
            .copied()
            .ok_or_else(invalid_sequence_boundaries)?;
        let image_start = vision_start
            .checked_add(1)
            .ok_or_else(invalid_sequence_boundaries)?;
        let image_end = image_start
            .checked_add(grid.tokens)
            .ok_or_else(invalid_sequence_boundaries)?;
        let image_run = input_tokens
            .get(image_start..image_end)
            .ok_or_else(invalid_sequence_boundaries)?;
        if image_run.iter().any(|token| *token != image_token_id)
            || input_tokens.get(image_end) == Some(&image_token_id)
        {
            return Err(invalid_sequence_boundaries());
        }

        let text_length = image_start - cursor;
        append_text_positions(&mut positions, text_length, next_position)?;
        let visual_start = next_position
            .checked_add(i64::try_from(text_length).map_err(|_| invalid_sequence_boundaries())?)
            .ok_or_else(invalid_sequence_boundaries)?;
        append_image_positions(&mut positions, grid, visual_start)?;
        let height_max =
            i64::try_from(grid.height - 1).map_err(|_| invalid_sequence_boundaries())?;
        let width_max = i64::try_from(grid.width - 1).map_err(|_| invalid_sequence_boundaries())?;
        next_position = visual_start
            .checked_add(height_max.max(width_max))
            .and_then(|position| position.checked_add(1))
            .ok_or_else(invalid_sequence_boundaries)?;
        cursor = image_end;
        grid_index += 1;
    }

    if grid_index != grids.len()
        || input_tokens[cursor..]
            .iter()
            .any(|token| *token == image_token_id || *token == vision_start_token_id)
    {
        return Err(invalid_sequence_boundaries());
    }
    append_text_positions(&mut positions, input_tokens.len() - cursor, next_position)?;
    if positions
        .iter()
        .any(|axis| axis.len() != input_tokens.len())
    {
        return Err(invalid_sequence_boundaries());
    }
    Ok(positions)
}

fn append_text_positions(
    positions: &mut [Vec<i64>; 3],
    length: usize,
    start: i64,
) -> Result<(), CpuRefError> {
    for offset in 0..length {
        let offset = i64::try_from(offset).map_err(|_| invalid_sequence_boundaries())?;
        let position = start
            .checked_add(offset)
            .ok_or_else(invalid_sequence_boundaries)?;
        for axis in positions.iter_mut() {
            axis.push(position);
        }
    }
    Ok(())
}

fn append_image_positions(
    positions: &mut [Vec<i64>; 3],
    grid: MergedImageGrid,
    start: i64,
) -> Result<(), CpuRefError> {
    for _ in 0..grid.temporal {
        for height in 0..grid.height {
            let height = start
                .checked_add(i64::try_from(height).map_err(|_| invalid_sequence_boundaries())?)
                .ok_or_else(invalid_sequence_boundaries)?;
            for width in 0..grid.width {
                let width = start
                    .checked_add(i64::try_from(width).map_err(|_| invalid_sequence_boundaries())?)
                    .ok_or_else(invalid_sequence_boundaries)?;
                positions[0].push(start);
                positions[1].push(height);
                positions[2].push(width);
            }
        }
    }
    Ok(())
}

fn checked_rope_delta(max_position: i64, input_length: usize) -> Result<i64, CpuRefError> {
    let input_length = i64::try_from(input_length).map_err(|_| invalid_sequence_boundaries())?;
    max_position
        .checked_add(1)
        .and_then(|position| position.checked_sub(input_length))
        .ok_or_else(invalid_sequence_boundaries)
}

const fn invalid_sequence_boundaries() -> CpuRefError {
    CpuRefError::new(
        CpuRefErrorCode::InvalidSequenceBoundaries,
        "multimodal token, mask, or position boundaries are invalid",
    )
}
