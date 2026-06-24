package tech.kayys.tafkir.ml.gguf;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

/**
 * GGUF metadata value types.
 * Extended to support all GGUF metadata types.
 */
public sealed interface GgufMetaValue extends Serializable
        permits GgufMetaValue.StringVal,
        GgufMetaValue.Uint8Val, GgufMetaValue.Int8Val,
        GgufMetaValue.Uint16Val, GgufMetaValue.Int16Val,
        GgufMetaValue.Uint32Val, GgufMetaValue.Int32Val,
        GgufMetaValue.Float32Val,
        GgufMetaValue.Uint64Val, GgufMetaValue.Int64Val,
        GgufMetaValue.Float64Val,
        GgufMetaValue.BoolVal,
        GgufMetaValue.ArrayVal {

    @Serial
    long serialVersionUID = 1L;

    default GgufMetaType getType() { throw new UnsupportedOperationException(); }
    default byte getUint8Value() { throw new UnsupportedOperationException(); }
    default byte getInt8Value() { throw new UnsupportedOperationException(); }
    default short getUint16Value() { throw new UnsupportedOperationException(); }
    default short getInt16Value() { throw new UnsupportedOperationException(); }
    default int getUint32Value() { throw new UnsupportedOperationException(); }
    default int getInt32Value() { throw new UnsupportedOperationException(); }
    default float getFloat32Value() { throw new UnsupportedOperationException(); }
    default long getUint64Value() { throw new UnsupportedOperationException(); }
    default long getInt64Value() { throw new UnsupportedOperationException(); }
    default double getFloat64Value() { throw new UnsupportedOperationException(); }
    default boolean getBoolValue() { throw new UnsupportedOperationException(); }
    default String getStringValue() { throw new UnsupportedOperationException(); }
    default List<GgufMetaValue> getArrayValue() { throw new UnsupportedOperationException(); }

    record StringVal(String value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.STRING; }
        public String getStringValue() { return value; }
    }
    record Uint8Val(byte value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.UINT8; }
        public byte getUint8Value() { return value; }
    }
    record Int8Val(byte value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.INT8; }
        public byte getInt8Value() { return value; }
    }
    record Uint16Val(short value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.UINT16; }
        public short getUint16Value() { return value; }
    }
    record Int16Val(short value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.INT16; }
        public short getInt16Value() { return value; }
    }
    record Uint32Val(long value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.UINT32; }
        public int getUint32Value() { return (int) value; }
    }
    record Int32Val(int value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.INT32; }
        public int getInt32Value() { return value; }
    }
    record Float32Val(float value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.FLOAT32; }
        public float getFloat32Value() { return value; }
    }
    record Uint64Val(long value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.UINT64; }
        public long getUint64Value() { return value; }
    }
    record Int64Val(long value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.INT64; }
        public long getInt64Value() { return value; }
    }
    record Float64Val(double value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.FLOAT64; }
        public double getFloat64Value() { return value; }
    }
    record BoolVal(boolean value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.BOOL; }
        public boolean getBoolValue() { return value; }
    }
    record ArrayVal(List<GgufMetaValue> value) implements GgufMetaValue {
        public GgufMetaType getType() { return GgufMetaType.ARRAY; }
        public List<GgufMetaValue> getArrayValue() { return value; }
    }

    static GgufMetaValue ofString(String value) { return new StringVal(value); }
    static GgufMetaValue ofUint32(long value) { return new Uint32Val(value); }
    static GgufMetaValue ofInt32(int value) { return new Int32Val(value); }
    static GgufMetaValue ofFloat32(float value) { return new Float32Val(value); }
    static GgufMetaValue ofBool(boolean value) { return new BoolVal(value); }
    static GgufMetaValue ofArray(List<GgufMetaValue> values) { return new ArrayVal(values); }
    static GgufMetaValue ofArray(GgufMetaValue... values) { return new ArrayVal(List.of(values)); }

    static GgufMetaValue of(byte value) { return new Int8Val(value); } // Simplified for reader compatibility
    static GgufMetaValue of(short value) { return new Int16Val(value); }
    static GgufMetaValue of(int value) { return new Int32Val(value); }
    static GgufMetaValue of(long value) { return new Int64Val(value); }
    static GgufMetaValue of(float value) { return new Float32Val(value); }
    static GgufMetaValue of(double value) { return new Float64Val(value); }
    static GgufMetaValue of(boolean value) { return new BoolVal(value); }
    static GgufMetaValue of(String value) { return new StringVal(value); }
}