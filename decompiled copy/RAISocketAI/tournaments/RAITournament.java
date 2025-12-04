/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.core.AI
 *  ai.core.AIWithComputationBudget
 *  ai.core.ContinuingAI
 *  ai.core.InterruptibleAI
 *  rts.GameState
 *  rts.PartiallyObservableGameState
 *  rts.PhysicalGameState
 *  rts.PlayerAction
 *  rts.Trace
 *  rts.TraceEntry
 *  rts.units.UnitTypeTable
 *  util.XMLWriter
 */
package tournaments;

import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ContinuingAI;
import ai.core.InterruptibleAI;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.units.UnitTypeTable;
import util.XMLWriter;

class RAITournament {
    private static int TIMEOUT_CHECK_TOLERANCE = 20;
    private static boolean USE_CONTINUING_ON_INTERRUPTIBLE = true;
    List<AI> AIs;
    List<AI> opponentAIs;
    private int[][] wins;
    private int[][] ties;
    private int[][] AIcrashes;
    private int[][] opponentAIcrashes;
    private int[][] AItimeout;
    private int[][] opponentAItimeout;
    private double[][] accumTime;

    RAITournament(List<AI> list, List<AI> list2) {
        this.AIs = list;
        this.opponentAIs = list2;
        this.wins = new int[list.size()][list2.size()];
        this.ties = new int[list.size()][list2.size()];
        this.AIcrashes = new int[list.size()][list2.size()];
        this.opponentAIcrashes = new int[list.size()][list2.size()];
        this.AItimeout = new int[list.size()][list2.size()];
        this.opponentAItimeout = new int[list.size()][list2.size()];
        this.accumTime = new double[list.size()][list2.size()];
    }

    RAITournament(List<AI> list) {
        this(list, list);
    }

    void playSingleGame(int n, int n2, int n3, long l, long l2, boolean bl, boolean bl2, boolean bl3, boolean bl4, UnitTypeTable unitTypeTable, String string, Writer writer, Writer writer2, String[] stringArray, boolean[][] blArray, int n4, int n5, PhysicalGameState physicalGameState, int n6, int n7) throws Exception {
        PlayerAction playerAction;
        Object object;
        TraceEntry traceEntry;
        int n8 = 0;
        int n9 = 0;
        double d = 0.0;
        double d2 = 0.0;
        int n10 = 0;
        int n11 = 0;
        double d3 = 0.0;
        double d4 = 0.0;
        int n12 = 0;
        int n13 = 0;
        double d5 = 0.0;
        double d6 = 0.0;
        AI aI = this.AIs.get(n6).clone();
        AI aI2 = this.opponentAIs.get(n7).clone();
        if (aI instanceof AIWithComputationBudget) {
            ((AIWithComputationBudget)aI).setTimeBudget(n2);
            ((AIWithComputationBudget)aI).setIterationsBudget(n3);
        }
        if (aI2 instanceof AIWithComputationBudget) {
            ((AIWithComputationBudget)aI2).setTimeBudget(n2);
            ((AIWithComputationBudget)aI2).setIterationsBudget(n3);
        }
        if (USE_CONTINUING_ON_INTERRUPTIBLE) {
            if (aI instanceof InterruptibleAI) {
                aI = new ContinuingAI(aI);
            }
            if (aI2 instanceof InterruptibleAI) {
                aI2 = new ContinuingAI(aI2);
            }
        }
        aI.reset();
        aI2.reset();
        GameState gameState = new GameState(physicalGameState.clone(), unitTypeTable);
        if (writer2 != null) {
            writer2.write("MATCH UP: " + String.valueOf(aI) + " vs " + String.valueOf(aI2) + "\n");
        }
        if (bl4 && blArray != null) {
            RAITournament.preAnalysisSingleAI(l, l2, writer2, stringArray[n6], blArray[n6], n5, aI, gameState);
            RAITournament.preAnalysisSingleAI(l, l2, writer2, stringArray[n7], blArray[n7], n5, aI2, gameState);
        }
        boolean bl5 = false;
        int n14 = -1;
        int n15 = -1;
        Trace trace = null;
        if (string != null) {
            trace = new Trace(unitTypeTable);
            traceEntry = new TraceEntry(gameState.getPhysicalGameState().clone(), gameState.getTime());
            trace.addEntry(traceEntry);
        }
        do {
            object = null;
            playerAction = null;
            long l3 = 0L;
            long l4 = 0L;
            long l5 = 0L;
            long l6 = 0L;
            if (bl3) {
                System.gc();
            }
            try {
                l3 = System.currentTimeMillis();
                object = aI.getAction(0, (GameState)(bl ? gameState : new PartiallyObservableGameState(gameState, 0)));
                l5 = System.currentTimeMillis();
            }
            catch (Exception exception) {
                if (writer2 != null) {
                    writer2.write(String.valueOf(exception) + "\n");
                    writer2.write(Arrays.toString(exception.getStackTrace()) + "\n");
                }
                n14 = 0;
                break;
            }
            if (bl3) {
                System.gc();
            }
            try {
                l4 = System.currentTimeMillis();
                playerAction = aI2.getAction(1, (GameState)(bl ? gameState : new PartiallyObservableGameState(gameState, 1)));
                l6 = System.currentTimeMillis();
            }
            catch (Exception exception) {
                if (writer2 != null) {
                    writer2.write(String.valueOf(exception) + "\n");
                    writer2.write(Arrays.toString(exception.getStackTrace()) + "\n");
                }
                n14 = 1;
                break;
            }
            long l7 = l5 - l3;
            long l8 = l6 - l4;
            ++n8;
            ++n9;
            d += (double)l7;
            d2 += (double)l8;
            if (l7 > (long)n2) {
                ++n10;
                d3 += (double)l7;
                if (l7 > (long)(n2 * 2)) {
                    ++n12;
                    d5 += (double)l7;
                }
            }
            if (l8 > (long)n2) {
                ++n11;
                d4 += (double)l8;
                if (l8 > (long)(n2 * 2)) {
                    ++n13;
                    d6 += (double)l8;
                }
            }
            if (bl2) {
                if (l7 > (long)(n2 + TIMEOUT_CHECK_TOLERANCE)) {
                    n15 = 0;
                    break;
                }
                if (l8 > (long)(n2 + TIMEOUT_CHECK_TOLERANCE)) {
                    n15 = 1;
                    break;
                }
            }
            if (!(string == null || object.isEmpty() && playerAction.isEmpty())) {
                traceEntry = new TraceEntry(gameState.getPhysicalGameState().clone(), gameState.getTime());
                traceEntry.addPlayerAction(object.clone());
                traceEntry.addPlayerAction(playerAction.clone());
                trace.addEntry(traceEntry);
            }
            gameState.issueSafe((PlayerAction)object);
            gameState.issueSafe(playerAction);
        } while (!(bl5 = gameState.cycle()) && gameState.getTime() < n);
        if (string != null) {
            object = new File(string);
            if (!((File)object).exists()) {
                ((File)object).mkdirs();
            }
            traceEntry = new TraceEntry(gameState.getPhysicalGameState().clone(), gameState.getTime());
            trace.addEntry(traceEntry);
            ZipOutputStream zipOutputStream = null;
            Object object2 = n6 + "-vs-" + n7 + "-" + n5 + "-" + n4;
            object2 = ((String)object2).replace("/", "");
            object2 = ((String)object2).replace(")", "");
            object2 = ((String)object2).replace("(", "");
            object2 = string + "/" + (String)object2;
            zipOutputStream = new ZipOutputStream(new FileOutputStream((String)object2 + ".zip"));
            zipOutputStream.putNextEntry(new ZipEntry("game.xml"));
            playerAction = new XMLWriter((Writer)new OutputStreamWriter(zipOutputStream));
            trace.toxml((XMLWriter)playerAction);
            playerAction.flush();
            zipOutputStream.closeEntry();
            zipOutputStream.close();
        }
        int n16 = -1;
        if (n14 != -1) {
            n16 = 1 - n14;
            if (n14 == 0) {
                int[] nArray = this.AIcrashes[n6];
                int n17 = n7;
                nArray[n17] = nArray[n17] + 1;
            }
            if (n14 == 1) {
                int[] nArray = this.opponentAIcrashes[n6];
                int n18 = n7;
                nArray[n18] = nArray[n18] + 1;
            }
        } else if (n15 != -1) {
            n16 = 1 - n15;
            if (n15 == 0) {
                int[] nArray = this.AItimeout[n6];
                int n19 = n7;
                nArray[n19] = nArray[n19] + 1;
            }
            if (n15 == 1) {
                int[] nArray = this.opponentAItimeout[n6];
                int n20 = n7;
                nArray[n20] = nArray[n20] + 1;
            }
        } else {
            n16 = gameState.winner();
        }
        aI.gameOver(n16);
        aI2.gameOver(n16);
        double d7 = d / (double)n8;
        double d8 = d2 / (double)n9;
        writer.write(n4 + "\t" + n5 + "\t" + n6 + "\t" + n7 + "\t" + gameState.getTime() + "\t" + n16 + "\t" + n14 + "\t" + n15 + "\t" + d7 + "\t" + d8 + "\t" + n10 + "\t" + n11 + "\n");
        writer.flush();
        if (writer2 != null) {
            writer2.write("Winner: " + n16 + "  in " + gameState.getTime() + " cycles\n");
            writer2.write(String.valueOf(aI) + " : " + aI.statisticsString() + "\n");
            writer2.write(String.valueOf(aI2) + " : " + aI2.statisticsString() + "\n");
            writer2.write("AI1 time usage, average:  " + d7 + ", # times over budget: " + n10 + " (avg " + d3 / (double)n10 + ") , # times over 2*budget: " + n12 + " (avg " + d5 / (double)n12 + ")\n");
            writer2.write("AI2 time usage, average:  " + d8 + ", # times over budget: " + n11 + " (avg " + d4 / (double)n11 + ") , # times over 2*budget: " + n13 + " (avg " + d6 / (double)n13 + ")\n");
            writer2.flush();
        }
        if (n16 == -1) {
            int[] nArray = this.ties[n6];
            int n21 = n7;
            nArray[n21] = nArray[n21] + 1;
        } else if (n16 == 0) {
            int[] nArray = this.wins[n6];
            int n22 = n7;
            nArray[n22] = nArray[n22] + 1;
        } else if (n16 == 1) {
            // empty if block
        }
        double[] dArray = this.accumTime[n6];
        int n23 = n7;
        dArray[n23] = dArray[n23] + (double)gameState.getTime();
    }

    private static void preAnalysisSingleAI(long l, long l2, Writer writer, String string, boolean[] blArray, int n, AI aI, GameState gameState) throws Exception {
        long l3 = l2;
        if (blArray[n]) {
            l3 = l;
            blArray[n] = false;
        }
        long l4 = System.currentTimeMillis();
        aI.preGameAnalysis(gameState, l3, string);
        long l5 = System.currentTimeMillis();
        if (writer != null) {
            writer.write("preGameAnalysis player 1 took " + (l5 - l4) + "\n");
            if (l5 - l4 > l3) {
                writer.write("TIMEOUT PLAYER 1!\n");
            }
        }
    }

    void printEndSummary(List<String> list, int n, Writer writer, Writer writer2) throws IOException {
        int n2;
        int n3;
        writer.write("Wins:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.wins[n3][n2] + "\t");
            }
            writer.write("\n");
        }
        writer.write("Ties:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.ties[n3][n2] + "\t");
            }
            writer.write("\n");
        }
        writer.write("Average Game Length:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.accumTime[n3][n2] / (double)(list.size() * n) + "\t");
            }
            writer.write("\n");
        }
        writer.write("AI crashes:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.AIcrashes[n3][n2] + "\t");
            }
            writer.write("\n");
        }
        writer.write("opponent AI crashes:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.opponentAIcrashes[n3][n2] + "\t");
            }
            writer.write("\n");
        }
        writer.write("AI timeout:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.AItimeout[n3][n2] + "\t");
            }
            writer.write("\n");
        }
        writer.write("opponent AI timeout:\n");
        for (n3 = 0; n3 < this.AIs.size(); ++n3) {
            for (n2 = 0; n2 < this.opponentAIs.size(); ++n2) {
                writer.write(this.opponentAItimeout[n3][n2] + "\t");
            }
            writer.write("\n");
        }
        writer.flush();
        if (writer2 != null) {
            writer2.write(this.getClass().getName() + ": tournament ended\n");
        }
        writer2.flush();
    }
}

