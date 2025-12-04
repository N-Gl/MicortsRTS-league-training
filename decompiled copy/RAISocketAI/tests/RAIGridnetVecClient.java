/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  ai.reward.RewardFunctionInterface
 *  rts.units.UnitTypeTable
 */
package tests;

import ai.core.AI;
import ai.rai.RAIResponse;
import ai.rai.RAIResponses;
import ai.reward.RewardFunctionInterface;
import java.util.Arrays;
import rts.units.UnitTypeTable;
import tests.RAIGridnetClient;
import tests.RAIGridnetClientSelfPlay;

public class RAIGridnetVecClient {
    public RAIGridnetClient[] clients;
    public RAIGridnetClientSelfPlay[] selfPlayClients;
    public int maxSteps;
    public int[] envSteps;
    public RewardFunctionInterface[] rfs;
    public UnitTypeTable utt;
    boolean partialObs = false;
    public String[] mapPaths;
    byte[][] mask;
    byte[][] observation;
    double[][] reward;
    boolean[][] done;
    byte[][] resources;
    RAIResponse[] rs;
    RAIResponses responses;
    byte[][] terrain;
    double[] terminalReward1;
    boolean[] terminalDone1;
    double[] terminalReward2;
    boolean[] terminalDone2;

    public RAIGridnetVecClient(int n, int n2, int n3, RewardFunctionInterface[] rewardFunctionInterfaceArray, String string, String[] stringArray, AI[] aIArray, UnitTypeTable unitTypeTable, boolean bl) throws Exception {
        int n4;
        this.maxSteps = n3;
        this.utt = unitTypeTable;
        this.rfs = rewardFunctionInterfaceArray;
        this.partialObs = bl;
        this.mapPaths = stringArray;
        this.envSteps = new int[n + n2];
        this.selfPlayClients = new RAIGridnetClientSelfPlay[n / 2];
        for (n4 = 0; n4 < this.selfPlayClients.length; ++n4) {
            this.selfPlayClients[n4] = new RAIGridnetClientSelfPlay(rewardFunctionInterfaceArray, string, this.mapPaths[n4 * 2], unitTypeTable, this.partialObs);
        }
        this.clients = new RAIGridnetClient[n2];
        for (n4 = 0; n4 < this.clients.length; ++n4) {
            this.clients[n4] = new RAIGridnetClient(rewardFunctionInterfaceArray, string, this.mapPaths[n + n4], aIArray[n4], unitTypeTable, this.partialObs);
        }
        n4 = n + n2;
        this.mask = new byte[n4][];
        this.observation = new byte[n4][];
        this.reward = new double[n4][this.rfs.length];
        this.done = new boolean[n4][this.rfs.length];
        this.resources = new byte[n4][2];
        this.terminalReward1 = new double[this.rfs.length];
        this.terminalDone1 = new boolean[this.rfs.length];
        this.terminalReward2 = new double[this.rfs.length];
        this.terminalDone2 = new boolean[this.rfs.length];
        this.responses = new RAIResponses(null, null, null, null, null, null);
        this.terrain = new byte[n4][];
        this.rs = new RAIResponse[n4];
    }

    public RAIResponses reset(int[] nArray) throws Exception {
        int n;
        for (n = 0; n < this.selfPlayClients.length; ++n) {
            this.selfPlayClients[n].reset();
            for (int i = 0; i < 2; ++i) {
                this.rs[n * 2 + i] = this.selfPlayClients[n].getResponse(i);
            }
        }
        Arrays.fill(this.envSteps, 0);
        for (n = this.selfPlayClients.length * 2; n < nArray.length; ++n) {
            this.rs[n] = this.clients[n - this.selfPlayClients.length * 2].reset(nArray[n]);
        }
        for (n = 0; n < this.rs.length; ++n) {
            this.observation[n] = this.rs[n].observation;
            this.mask[n] = this.rs[n].mask;
            this.reward[n] = this.rs[n].reward;
            this.done[n] = this.rs[n].done;
            this.terrain[n] = this.rs[n].terrain;
            this.resources[n] = this.rs[n].resources;
        }
        this.responses.set(this.observation, this.mask, this.reward, this.done, this.terrain, this.resources);
        return this.responses;
    }

    public RAIResponses gameStep(int[][][] nArray, int[] nArray2) throws Exception {
        int n;
        int n2;
        for (n2 = 0; n2 < this.selfPlayClients.length; ++n2) {
            this.selfPlayClients[n2].gameStep(nArray[n2 * 2], nArray[n2 * 2 + 1]);
            this.rs[n2 * 2] = this.selfPlayClients[n2].getResponse(0);
            this.rs[n2 * 2 + 1] = this.selfPlayClients[n2].getResponse(1);
            int n3 = n2 * 2;
            this.envSteps[n3] = this.envSteps[n3] + 1;
            int n4 = n2 * 2 + 1;
            this.envSteps[n4] = this.envSteps[n4] + 1;
            if (!this.rs[n2 * 2].done[0] && this.envSteps[n2 * 2] < this.maxSteps) continue;
            for (n = 0; n < this.terminalReward1.length; ++n) {
                this.terminalReward1[n] = this.rs[n2 * 2].reward[n];
                this.terminalDone1[n] = this.rs[n2 * 2].done[n];
                this.terminalReward2[n] = this.rs[n2 * 2 + 1].reward[n];
                this.terminalDone2[n] = this.rs[n2 * 2 + 1].done[n];
            }
            this.selfPlayClients[n2].reset();
            for (n = 0; n < this.terminalReward1.length; ++n) {
                this.rs[n2 * 2].reward[n] = this.terminalReward1[n];
                this.rs[n2 * 2].done[n] = this.terminalDone1[n];
                this.rs[n2 * 2 + 1].reward[n] = this.terminalReward2[n];
                this.rs[n2 * 2 + 1].done[n] = this.terminalDone2[n];
            }
            this.rs[n2 * 2].done[0] = true;
            this.rs[n2 * 2 + 1].done[0] = true;
            this.envSteps[n2 * 2] = 0;
            this.envSteps[n2 * 2 + 1] = 0;
        }
        for (n2 = this.selfPlayClients.length * 2; n2 < nArray2.length; ++n2) {
            int n5 = n2;
            this.envSteps[n5] = this.envSteps[n5] + 1;
            this.rs[n2] = this.clients[n2 - this.selfPlayClients.length * 2].gameStep(nArray[n2], nArray2[n2]);
            if (!this.rs[n2].done[0] && this.envSteps[n2] < this.maxSteps) continue;
            for (n = 0; n < this.rs[n2].reward.length; ++n) {
                this.terminalReward1[n] = this.rs[n2].reward[n];
                this.terminalDone1[n] = this.rs[n2].done[n];
            }
            this.clients[n2 - this.selfPlayClients.length * 2].reset(nArray2[n2]);
            for (n = 0; n < this.rs[n2].reward.length; ++n) {
                this.rs[n2].reward[n] = this.terminalReward1[n];
                this.rs[n2].done[n] = this.terminalDone1[n];
            }
            this.rs[n2].done[0] = true;
            this.envSteps[n2] = 0;
        }
        for (n2 = 0; n2 < this.rs.length; ++n2) {
            this.observation[n2] = this.rs[n2].observation;
            this.mask[n2] = this.rs[n2].mask;
            this.reward[n2] = this.rs[n2].reward;
            this.done[n2] = this.rs[n2].done;
            this.resources[n2] = this.rs[n2].resources;
        }
        this.responses.set(this.observation, this.mask, this.reward, this.done, null, this.resources);
        return this.responses;
    }

    public void close() throws Exception {
        if (this.clients != null) {
            for (RAIGridnetClient object : this.clients) {
                object.close();
            }
        }
        if (this.selfPlayClients != null) {
            for (RAIGridnetClientSelfPlay rAIGridnetClientSelfPlay : this.selfPlayClients) {
                rAIGridnetClientSelfPlay.close();
            }
        }
    }
}

