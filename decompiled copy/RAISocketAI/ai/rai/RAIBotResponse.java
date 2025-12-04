/*
 * Decompiled with CFR 0.152.
 */
package ai.rai;

import ai.rai.RAIResponse;

public class RAIBotResponse
extends RAIResponse {
    public int[][] action;

    public RAIBotResponse(byte[] byArray, byte[] byArray2, double[] dArray, boolean[] blArray, String string, byte[] byArray3, byte[] byArray4, int[][] nArray) {
        super(byArray, byArray2, dArray, blArray, string, byArray3, byArray4);
        this.action = nArray;
    }

    public void set(byte[] byArray, byte[] byArray2, double[] dArray, boolean[] blArray, String string, byte[] byArray3, byte[] byArray4, int[][] nArray) {
        super.set(byArray, byArray2, dArray, blArray, string, byArray3, byArray4);
        this.action = nArray;
    }
}

