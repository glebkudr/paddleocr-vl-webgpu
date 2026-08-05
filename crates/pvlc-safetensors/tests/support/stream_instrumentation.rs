use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};

#[derive(Debug)]
pub(crate) struct InstrumentedReader {
    inner: Cursor<Vec<u8>>,
    forbid_reads_at_or_after: Option<u64>,
    pub(crate) max_requested_read: usize,
    pub(crate) total_read: u64,
}

impl InstrumentedReader {
    pub(crate) fn new(bytes: Vec<u8>, forbid_reads_at_or_after: Option<u64>) -> Self {
        Self {
            inner: Cursor::new(bytes),
            forbid_reads_at_or_after,
            max_requested_read: 0,
            total_read: 0,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn allow_all_reads_and_reset_metrics(&mut self) {
        self.forbid_reads_at_or_after = None;
        self.max_requested_read = 0;
        self.total_read = 0;
    }
}

impl Read for InstrumentedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.max_requested_read = self.max_requested_read.max(buffer.len());
        let position = self.inner.position();
        let allowed = match self.forbid_reads_at_or_after {
            Some(boundary) if position >= boundary => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "test reader forbids payload reads",
                ));
            }
            Some(boundary) => buffer.len().min((boundary - position) as usize),
            None => buffer.len(),
        };
        let read = self.inner.read(&mut buffer[..allowed])?;
        self.total_read += read as u64;
        Ok(read)
    }
}

impl Seek for InstrumentedReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

#[derive(Debug, Default)]
pub(crate) struct PartialThenFailWriter {
    pub(crate) calls: usize,
    pub(crate) accepted: Vec<u8>,
}

impl Write for PartialThenFailWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.calls >= 2 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "intentional writer backpressure failure",
            ));
        }
        let accepted = buffer.len().min(97);
        self.accepted.extend_from_slice(&buffer[..accepted]);
        Ok(accepted)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
