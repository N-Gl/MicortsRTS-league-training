/*
 * Decompiled with CFR 0.152.
 */
package ai.rai;

public class RAIResponses {
    public byte[][] observation;
    public byte[][] mask;
    public double[][] reward;
    public boolean[][] done;
    public byte[][] terrain;
    public byte[][] resources;

    public RAIResponses(byte[][] byArray, byte[][] byArray2, double[][] dArray, boolean[][] blArray, byte[][] byArray3, byte[][] byArray4) {
        this.observation = byArray;
        this.mask = byArray2;
        this.reward = dArray;
        this.done = blArray;
        this.terrain = byArray3;
        this.resources = byArray4;
    }

    public void set(byte[][] byArray, byte[][] byArray2, double[][] dArray, boolean[][] blArray, byte[][] byArray3, byte[][] byArray4) {
        this.observation = byArray;
        this.mask = byArray2;
        this.reward = dArray;
        this.done = blArray;
        this.terrain = byArray3;
        this.resources = byArray4;
    }
}

