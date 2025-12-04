/*
 * Decompiled with CFR 0.152.
 * 
 * Could not load the following classes:
 *  ai.coac.CoacAI
 *  gui.PhysicalGameStateJFrame
 *  gui.PhysicalGameStatePanel
 *  rts.GameState
 *  rts.PhysicalGameState
 *  rts.PlayerAction
 *  rts.units.UnitTypeTable
 */
package tests;

import ai.coac.CoacAI;
import ai.rai.RAISocketAI;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class RAIGameVisualSimulationTest {
    public static void main(String[] stringArray) throws Exception {
        UnitTypeTable unitTypeTable = new UnitTypeTable(2);
        RAISocketAI rAISocketAI = new RAISocketAI(unitTypeTable);
        CoacAI coacAI = new CoacAI(unitTypeTable);
        PhysicalGameState physicalGameState = PhysicalGameState.load((String)"maps/16x16/TwoBasesBarracks16x16.xml", (UnitTypeTable)unitTypeTable);
        GameState gameState = new GameState(physicalGameState, unitTypeTable);
        int n = 4000;
        int n2 = 20;
        boolean bl = false;
        PhysicalGameStateJFrame physicalGameStateJFrame = PhysicalGameStatePanel.newVisualizer((GameState)gameState, (int)640, (int)640, (boolean)false, (int)PhysicalGameStatePanel.COLORSCHEME_BLACK);
        long l = System.currentTimeMillis() + (long)n2;
        do {
            if (System.currentTimeMillis() >= l) {
                PlayerAction playerAction = rAISocketAI.getAction(0, gameState);
                PlayerAction playerAction2 = coacAI.getAction(1, gameState);
                gameState.issueSafe(playerAction);
                gameState.issueSafe(playerAction2);
                bl = gameState.cycle();
                physicalGameStateJFrame.repaint();
                l += (long)n2;
                continue;
            }
            try {
                Thread.sleep(1L);
            }
            catch (Exception exception) {
                exception.printStackTrace();
            }
        } while (!bl && gameState.getTime() < n);
        rAISocketAI.gameOver(gameState.winner());
        coacAI.gameOver(gameState.winner());
        physicalGameStateJFrame.dispose();
        System.out.println("Game Over");
    }
}

