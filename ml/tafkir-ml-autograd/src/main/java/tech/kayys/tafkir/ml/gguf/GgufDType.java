package tech.kayys.tafkir.ml.gguf;

import java.util.*;

/**
 * GGUF data types for tensors.
 */
public enum GgufDType {
    F32(0),
    F16(1),
    Q4_0(2),
    Q4_1(3),
    Q5_0(6),
    Q5_1(7),
    Q8_0(8),
    Q8_1(9),
    Q2_K(10),
    Q3_K(11),
    Q4_K(12),
    Q5_K(13),
    Q6_K(14),
    Q8_K(15);

    private final int code;

    GgufDType(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    public static GgufDType fromCode(int code) {
        for (GgufDType type : values()) {
            if (type.code == code) {
                return type;
            }
        }
        return null;
    }
}