/*
 * Decompiled with CFR 0.152.
 */
package ai.rai;

import ai.rai.RAIResponses;

public class RAIBotResponses
extends RAIResponses {
    public int[][][] action;

    public RAIBotResponses(byte[][] byArray, byte[][] byArray2, double[][] dArray, boolean[][] blArray, byte[][] byArray3, byte[][] byArray4, int[][][] nArray) {
        super(byArray, byArray2, dArray, blArray, byArray3, byArray4);
        this.action = nArray;
    }

    public void set(byte[][] byArray, byte[][] byArray2, double[][] dArray, boolean[][] blArray, byte[][] byArray3, byte[][] byArray4, int[][][] nArray) {
        super.set(byArray, byArray2, dArray, blArray, byArray3, byArray4);
        this.action = nArray;
    }
}

