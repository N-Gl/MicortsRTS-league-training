/*
 * Decompiled with CFR 0.152.
 */
package ai.rai;

public class RAIResponse {
    public byte[] observation;
    public byte[] mask;
    public double[] reward;
    public boolean[] done;
    public String info;
    public byte[] terrain;
    public byte[] resources;

    public RAIResponse(byte[] byArray, byte[] byArray2, double[] dArray, boolean[] blArray, String string, byte[] byArray3, byte[] byArray4) {
        this.observation = byArray;
        this.mask = byArray2;
        this.reward = dArray;
        this.done = blArray;
        this.info = string;
        this.terrain = byArray3;
        this.resources = byArray4;
    }

    public void set(byte[] byArray, byte[] byArray2, double[] dArray, boolean[] blArray, String string, byte[] byArray3, byte[] byArray4) {
        this.observation = byArray;
        this.mask = byArray2;
        this.reward = dArray;
        this.done = blArray;
        this.info = string;
        this.terrain = byArray3;
        this.resources = byArray4;
    }
}

