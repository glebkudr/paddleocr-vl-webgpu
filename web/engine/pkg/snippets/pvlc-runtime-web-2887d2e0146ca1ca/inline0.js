
const PVLC_UINT8_ARRAY = Uint8Array;
const PVLC_UINT8_ARRAY_PROTOTYPE = PVLC_UINT8_ARRAY.prototype;
const PVLC_TYPED_ARRAY_CONSTRUCTOR = Object.getPrototypeOf(PVLC_UINT8_ARRAY);
const PVLC_TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(PVLC_UINT8_ARRAY_PROTOTYPE);
const PVLC_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const PVLC_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const PVLC_REFLECT_APPLY = Reflect.apply;
const PVLC_ERROR = Error;
const PVLC_GLOBAL = globalThis;
const PVLC_SYMBOL_SPECIES = Symbol.species;
const PVLC_UINT8_ARRAY_GLOBAL_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_GLOBAL, "Uint8Array");
const PVLC_UINT8_ARRAY_CONSTRUCTOR_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "constructor");
const PVLC_UINT8_ARRAY_SPECIES_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY, PVLC_SYMBOL_SPECIES);
const PVLC_TYPED_ARRAY_SPECIES_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
        PVLC_TYPED_ARRAY_CONSTRUCTOR,
        PVLC_SYMBOL_SPECIES,
    );
const PVLC_BRIDGE_PROPERTY_NAMES = [
    "byteLength",
    "length",
    "set",
    "subarray",
    "slice",
];
const PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS = [
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "byteLength"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "length"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "set"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "subarray"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "slice"),
];
const PVLC_UINT8_ARRAY_PROPERTY_DESCRIPTORS = [
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "byteLength"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "length"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "set"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "subarray"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "slice"),
];
const PVLC_TYPED_ARRAY_BYTE_LENGTH_GETTER =
    PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS[0]?.get;
const PVLC_TYPED_ARRAY_SET = PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS[2]?.value;

function pvlcOwnDescriptorField(descriptor, name) {
    const field = PVLC_GET_OWN_PROPERTY_DESCRIPTOR(descriptor, name);
    return field === undefined ? undefined : field.value;
}

function pvlcDescriptorMatches(expected, observed) {
    if (expected === undefined || observed === undefined) {
        return expected === observed;
    }
    return (
        pvlcOwnDescriptorField(expected, "value") ===
            pvlcOwnDescriptorField(observed, "value") &&
        pvlcOwnDescriptorField(expected, "get") ===
            pvlcOwnDescriptorField(observed, "get") &&
        pvlcOwnDescriptorField(expected, "set") ===
            pvlcOwnDescriptorField(observed, "set") &&
        pvlcOwnDescriptorField(expected, "configurable") ===
            pvlcOwnDescriptorField(observed, "configurable") &&
        pvlcOwnDescriptorField(expected, "enumerable") ===
            pvlcOwnDescriptorField(observed, "enumerable") &&
        pvlcOwnDescriptorField(expected, "writable") ===
            pvlcOwnDescriptorField(observed, "writable")
    );
}

function pvlcAssertUint8ArrayBridgeIntrinsics() {
    if (!pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_GLOBAL_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_GLOBAL, "Uint8Array"),
        ) ||
        PVLC_UINT8_ARRAY.prototype !== PVLC_UINT8_ARRAY_PROTOTYPE ||
        PVLC_GET_PROTOTYPE_OF(PVLC_UINT8_ARRAY) !==
            PVLC_TYPED_ARRAY_CONSTRUCTOR ||
        PVLC_GET_PROTOTYPE_OF(PVLC_UINT8_ARRAY_PROTOTYPE) !==
            PVLC_TYPED_ARRAY_PROTOTYPE ||
        !pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_CONSTRUCTOR_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
                PVLC_UINT8_ARRAY_PROTOTYPE,
                "constructor",
            ),
        ) ||
        !pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_SPECIES_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
                PVLC_UINT8_ARRAY,
                PVLC_SYMBOL_SPECIES,
            ),
        ) ||
        !pvlcDescriptorMatches(
            PVLC_TYPED_ARRAY_SPECIES_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
                PVLC_TYPED_ARRAY_CONSTRUCTOR,
                PVLC_SYMBOL_SPECIES,
            ),
        )) {
        throw new PVLC_ERROR("pvlc Uint8Array bridge intrinsic boundary drifted");
    }
    for (let index = 0; index < PVLC_BRIDGE_PROPERTY_NAMES.length; index += 1) {
        const name = PVLC_BRIDGE_PROPERTY_NAMES[index];
        const observedTypedArray =
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, name);
        const observedUint8Array =
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, name);
        if (!pvlcDescriptorMatches(
            PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS[index],
            observedTypedArray,
        ) || !pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_PROPERTY_DESCRIPTORS[index],
            observedUint8Array,
        )) {
            throw new PVLC_ERROR(
                `pvlc Uint8Array bridge intrinsic boundary drifted: ${name}`,
            );
        }
    }
}

export function own_pvlc_uint8array_bridge_input(value) {
    pvlcAssertUint8ArrayBridgeIntrinsics();
    try {
        if (PVLC_GET_PROTOTYPE_OF(value) !== PVLC_UINT8_ARRAY_PROTOTYPE ||
            typeof PVLC_TYPED_ARRAY_BYTE_LENGTH_GETTER !== "function" ||
            typeof PVLC_TYPED_ARRAY_SET !== "function") {
            throw new PVLC_ERROR("pvlc Uint8Array bridge input is invalid");
        }
        const byteLength = PVLC_REFLECT_APPLY(
            PVLC_TYPED_ARRAY_BYTE_LENGTH_GETTER,
            value,
            [],
        );
        const owned = new PVLC_UINT8_ARRAY(byteLength);
        PVLC_REFLECT_APPLY(PVLC_TYPED_ARRAY_SET, owned, [value, 0]);
        return owned;
    } catch {
        throw new PVLC_ERROR("pvlc Uint8Array bridge input is invalid");
    }
}
