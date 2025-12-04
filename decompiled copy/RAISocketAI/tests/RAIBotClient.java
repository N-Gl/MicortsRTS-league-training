/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  ai.reward.RewardFunctionInterface
 *  gui.PhysicalGameStateJFrame
 *  gui.PhysicalGameStatePanel
 *  rts.GameState
 *  rts.PartiallyObservableGameState
 *  rts.PhysicalGameState
 *  rts.PlayerAction
 *  rts.TraceEntry
 *  rts.units.Unit
 *  rts.units.UnitTypeTable
 */
package tests;

import ai.core.AI;
import ai.rai.GameStateWrapper;
import ai.rai.RAIBotResponse;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Paths;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.TraceEntry;
import rts.units.Unit;
import rts.units.UnitTypeTable;

public class RAIBotClient {
    PhysicalGameStateJFrame w;
    public AI[] ais;
    public PhysicalGameState pgs;
    public GameState gs;
    public GameState[] playergs = new GameState[2];
    UnitTypeTable utt;
    boolean partialObs;
    public RewardFunctionInterface[] rfs;
    public String mapPath;
    String micrortsPath;
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int maxAttackRadius;
    public final int numPlayers = 2;
    int[][][][] masks = new int[2][][][];
    double[][] rewards = new double[2][];
    boolean[][] dones = new boolean[2][];
    RAIBotResponse[] response = new RAIBotResponse[2];
    PlayerAction[] pas = new PlayerAction[2];

    public RAIBotClient(RewardFunctionInterface[] rewardFunctionInterfaceArray, String string, String string2, AI aI, AI aI2, UnitTypeTable unitTypeTable, boolean bl) throws Exception {
        this.micrortsPath = string;
        this.mapPath = string2;
        this.rfs = rewardFunctionInterfaceArray;
        this.utt = unitTypeTable;
        this.partialObs = bl;
        if (aI == null || aI2 == null) {
            throw new Exception("no ai1 or ai2 was chosen");
        }
        if (this.micrortsPath.length() != 0) {
            this.mapPath = Paths.get(this.micrortsPath, this.mapPath).toString();
        }
        this.pgs = PhysicalGameState.load((String)this.mapPath, (UnitTypeTable)this.utt);
        this.ais = new AI[]{aI, aI2};
        for (int i = 0; i < 2; ++i) {
            this.masks[i] = new int[this.pgs.getHeight()][this.pgs.getWidth()][23 + this.utt.getUnitTypes().size() + this.maxAttackRadius * this.maxAttackRadius];
            this.rewards[i] = new double[this.rfs.length];
            this.dones[i] = new boolean[this.rfs.length];
            this.response[i] = new RAIBotResponse(null, null, null, null, null, null, null, null);
        }
    }

    public byte[] render(boolean bl) throws Exception {
        if (this.w == null) {
            this.w = PhysicalGameStatePanel.newVisualizer((GameState)this.gs, (int)640, (int)640, (boolean)false, null, (int)this.renderTheme);
        }
        this.w.setStateCloning(this.gs);
        this.w.repaint();
        if (!bl) {
            return null;
        }
        BufferedImage bufferedImage = new BufferedImage(this.w.getWidth(), this.w.getHeight(), 5);
        this.w.paint(bufferedImage.getGraphics());
        WritableRaster writableRaster = bufferedImage.getRaster();
        DataBufferByte dataBufferByte = (DataBufferByte)writableRaster.getDataBuffer();
        return dataBufferByte.getData();
    }

    public void gameStep() throws Exception {
        int n;
        TraceEntry traceEntry = new TraceEntry(this.gs.getPhysicalGameState().clone(), this.gs.getTime());
        boolean bl = false;
        int[][][] nArrayArray = new int[2][][];
        for (n = 0; n < 2; ++n) {
            this.playergs[n] = this.gs;
            if (this.partialObs) {
                this.playergs[n] = new PartiallyObservableGameState(this.gs, n);
            }
            try {
                this.pas[n] = this.ais[n].getAction(n, this.playergs[n]);
            }
            catch (Throwable throwable) {
                Unit[] unitArray;
                System.err.println("Player " + n + " Error: " + throwable.getMessage());
                throwable.printStackTrace();
                int n2 = n;
                for (Unit unit2 : unitArray = (Unit[])this.gs.getUnits().stream().filter(unit -> unit.getPlayer() == n2).toArray(Unit[]::new)) {
                    this.gs.removeUnit(unit2);
                }
                bl = true;
                break;
            }
            this.gs.issueSafe(this.pas[n]);
            traceEntry.addPlayerAction(this.pas[n].clone());
            nArrayArray[n] = GameStateWrapper.toVectorAction(this.gs, this.pas[n]);
        }
        if (!bl) {
            this.gameover = this.gs.cycle();
        }
        if (this.gameover || bl) {
            for (n = 0; n < 2; ++n) {
                this.ais[n].gameOver(this.gs.winner());
            }
        }
        for (n = 0; n < 2; ++n) {
            for (int i = 0; i < this.rfs.length; ++i) {
                this.rfs[i].computeReward(n, 1 - n, traceEntry, this.gs);
                this.rewards[n][i] = this.rfs[i].getReward();
                this.dones[n][i] = this.rfs[i].isDone();
            }
            GameStateWrapper gameStateWrapper = new GameStateWrapper(this.gs, 0, true);
            this.response[n].set(gameStateWrapper.getArrayObservation(n), gameStateWrapper.getBinaryMask(n), this.rewards[n], this.dones[n], "{}", null, gameStateWrapper.getPlayerResources(n), nArrayArray[n]);
        }
    }

    public String sendUTT() throws Exception {
        StringWriter stringWriter = new StringWriter();
        this.utt.toJSON((Writer)stringWriter);
        return ((Object)stringWriter).toString();
    }

    public void reset() throws Exception {
        this.pgs = PhysicalGameState.load((String)this.mapPath, (UnitTypeTable)this.utt);
        this.gs = new GameState(this.pgs, this.utt);
        for (int i = 0; i < 2; ++i) {
            this.playergs[i] = this.gs;
            if (this.partialObs) {
                this.playergs[i] = new PartiallyObservableGameState(this.gs, i);
            }
            this.ais[i] = this.ais[i].clone();
            this.ais[i].reset();
            for (int j = 0; j < this.rewards.length; ++j) {
                this.rewards[i][j] = 0.0;
                this.dones[i][j] = false;
            }
            GameStateWrapper gameStateWrapper = new GameStateWrapper(this.gs, 0, true);
            this.response[i].set(gameStateWrapper.getArrayObservation(i), gameStateWrapper.getBinaryMask(i), this.rewards[i], this.dones[i], "{}", gameStateWrapper.getTerrain(), gameStateWrapper.getPlayerResources(i), null);
        }
    }

    public RAIBotResponse getResponse(int n) {
        return this.response[n];
    }

    public void close() throws Exception {
        if (this.w != null) {
            System.out.println(this.getClass().getSimpleName() + ": Not disposing frame. Resource Leak!");
        }
    }
}

