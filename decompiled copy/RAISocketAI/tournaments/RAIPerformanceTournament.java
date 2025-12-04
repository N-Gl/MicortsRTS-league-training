/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.coac.CoacAI
 *  ai.core.AI
 *  mayariBot.mayari
 *  rts.PhysicalGameState
 *  rts.units.UnitTypeTable
 *  util.Pair
 */
package tournaments;

import ai.coac.CoacAI;
import ai.core.AI;
import ai.rai.RAIBCPPOAI;
import ai.rai.RAISocketAI;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import mayariBot.mayari;
import rts.PhysicalGameState;
import rts.units.UnitTypeTable;
import tournaments.RAITournament;
import tournaments.RAITournamentArguments;
import util.Pair;

public class RAIPerformanceTournament
extends RAITournament {
    public RAIPerformanceTournament(List<AI> list) {
        super(list);
    }

    public static void main(String[] stringArray) throws Exception {
        int n;
        String string;
        Object object;
        RAITournamentArguments rAITournamentArguments = new RAITournamentArguments(stringArray);
        int n2 = rAITournamentArguments.getOptionInteger('p', 0);
        int n3 = rAITournamentArguments.pythonVerbosity;
        boolean bl = rAITournamentArguments.hasOption('b');
        String string2 = rAITournamentArguments.getOptionValue('m', "RAISocketAI");
        System.out.println("modelSet: " + string2);
        int n4 = rAITournamentArguments.getOptionInteger('t', 100);
        boolean bl2 = !rAITournamentArguments.hasOption('T');
        UnitTypeTable unitTypeTable = new UnitTypeTable(2);
        RAISocketAI rAISocketAI = string2.equals("RAI-BC-PPO") ? new RAIBCPPOAI(n4, -1, unitTypeTable, n2, n3, bl) : new RAISocketAI(n4, -1, unitTypeTable, n2, n3, bl, string2);
        AI[] aIArray = new AI[]{rAISocketAI, new CoacAI(unitTypeTable), new mayari(unitTypeTable)};
        RAIPerformanceTournament rAIPerformanceTournament = new RAIPerformanceTournament(Arrays.asList(aIArray));
        ArrayList<Pair> arrayList = new ArrayList<Pair>();
        arrayList.add(new Pair((Object)"maps/NoWhereToRun9x8.xml", (Object)5000));
        arrayList.add(new Pair((Object)"maps/16x16/TwoBasesBarracks16x16.xml", (Object)5000));
        arrayList.add(new Pair((Object)"maps/DoubleGame24x24.xml", (Object)6000));
        arrayList.add(new Pair((Object)"maps/BWDistantResources32x32.xml", (Object)7000));
        arrayList.add(new Pair((Object)"maps/BroodWar/(4)BloodBath.scmB.xml", (Object)10000));
        Object object2 = "tournament_";
        if (arrayList.size() == 1) {
            Path path = Paths.get((String)((Pair)arrayList.get((int)0)).m_a, new String[0]);
            object = path.getFileName().toString();
            string = ((String)object).substring(0, ((String)object).lastIndexOf("."));
            object2 = (String)object2 + string + "_";
        }
        int n5 = 0;
        while (((File)(object = new File((String)object2 + ++n5))).exists()) {
        }
        ((File)object).mkdir();
        string = ((File)object).getName();
        File file = new File(string + "/tournament.csv");
        String string3 = string + "/traces";
        String string4 = string;
        int n6 = rAITournamentArguments.getOptionInteger('n', 1);
        FileWriter fileWriter = new FileWriter(file);
        PrintWriter printWriter = new PrintWriter(System.out);
        if (printWriter != null) {
            ((Writer)printWriter).write(rAIPerformanceTournament.getClass().getName() + ": Starting tournament\n");
            ((Writer)printWriter).write("overrideTorchThreads: " + n2 + "; pythonVerboseLevel: " + n3);
            if (bl) {
                ((Writer)printWriter).write("; Use best models");
            }
            ((Writer)printWriter).write("\n");
        }
        fileWriter.write(rAIPerformanceTournament.getClass().getName() + "\n");
        fileWriter.write("AIs\n");
        for (AI aI : aIArray) {
            fileWriter.write("\t" + aI.toString() + "\n");
        }
        fileWriter.write("maps\n");
        for (Pair pair2 : arrayList) {
            fileWriter.write("\t" + (String)pair2.m_a + "\t" + String.valueOf(pair2.m_b) + "\n");
        }
        fileWriter.write("iterations\t" + n6 + "\n");
        fileWriter.write("timeBudget\t" + n4 + "\n");
        fileWriter.write("iterationsBudget\t-1\n");
        fileWriter.write("pregameAnalysisBudget\t10000\t1000\n");
        fileWriter.write("preAnalysis\ttrue\n");
        fileWriter.write("fullObservability\ttrue\n");
        fileWriter.write("timeoutCheck\t" + bl2 + "\n");
        fileWriter.write("runGC\ttrue\n");
        fileWriter.write("iteration\tmap\tai1\tai2\ttime\twinner\tcrashed\ttimedout\tai1time\tai2time\tai1over\tai2over\n");
        ((Writer)fileWriter).flush();
        String[] stringArray2 = new String[aIArray.length];
        for (int i = 0; i < aIArray.length; ++i) {
            stringArray2[i] = string4 + "/AI" + i + "readWriteFolder";
            File file2 = new File((String)stringArray2[i]);
            file2.mkdir();
        }
        boolean[][] blArray = new boolean[aIArray.length][arrayList.size()];
        for (n = 0; n < aIArray.length; ++n) {
            for (int i = 0; i < arrayList.size(); ++i) {
                blArray[n][i] = true;
            }
        }
        for (n = 0; n < n6; ++n) {
            for (int i = 0; i < arrayList.size(); ++i) {
                Pair pair3 = (Pair)arrayList.get(i);
                PhysicalGameState physicalGameState = PhysicalGameState.load((String)((String)pair3.m_a), (UnitTypeTable)unitTypeTable);
                for (int j = 0; j < aIArray.length; ++j) {
                    for (int k = 0; k < aIArray.length; ++k) {
                        if (j == k || j != 0 && k != 0) continue;
                        ((Writer)printWriter).write("Starting iteration " + n + " on " + (String)pair3.m_a + "\n");
                        ((Writer)printWriter).flush();
                        rAIPerformanceTournament.playSingleGame((Integer)pair3.m_b, n4, -1, 10000L, 1000L, true, bl2, true, true, unitTypeTable, string3, fileWriter, printWriter, stringArray2, blArray, n, i, physicalGameState, j, k);
                    }
                }
            }
        }
        rAIPerformanceTournament.printEndSummary(arrayList.stream().map(pair -> (String)pair.m_a).collect(Collectors.toList()), n6, fileWriter, printWriter);
    }
}

