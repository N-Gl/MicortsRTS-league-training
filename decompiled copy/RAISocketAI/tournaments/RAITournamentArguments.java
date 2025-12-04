/*
 * Decompiled with CFR 0.152.
 */
package tournaments;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class RAITournamentArguments {
    private CommandLine cmd;
    public int pythonVerbosity = 1;

    public RAITournamentArguments(String[] stringArray) throws ParseException {
        Options options = new Options();
        options.addOption("n", "num-iterations", true, "Number of iterations to play round-robin tournament");
        options.addOption("t", "time-budget", true, "Milliseconds each turn is allowed to have");
        options.addOption("T", "no-timeout", false, "Losses aren't declared for timeouts, but AIs still passed time-budget");
        options.addOption("b", "use-best-models", false, "Disable performance-based model selection in RAISocketAI. Always pick highest precedence model");
        options.addOption("p", "override-torch-threads", true, "Override torch threads to this value. Ignoring other logic");
        options.addOption("v", "python-verbose", false, "Make Python process logging extra verbose");
        options.addOption("q", "quiet", false, "Make Python process not log to file");
        options.addOption("m", "model-set", true, "Use the specified model set");
        DefaultParser defaultParser = new DefaultParser();
        try {
            this.cmd = defaultParser.parse(options, stringArray);
        }
        catch (ParseException parseException) {
            HelpFormatter helpFormatter = new HelpFormatter();
            helpFormatter.printHelp(StackWalker.getInstance(StackWalker.Option.RETAIN_CLASS_REFERENCE).getCallerClass().getSimpleName(), options);
            throw parseException;
        }
        if (this.cmd.hasOption('q')) {
            this.pythonVerbosity = 0;
        } else if (this.cmd.hasOption('v')) {
            this.pythonVerbosity = 2;
        }
    }

    public boolean hasOption(char c) {
        return this.cmd.hasOption(c);
    }

    public int getOptionInteger(char c, int n) {
        if (this.cmd.hasOption(c)) {
            return Integer.valueOf(this.cmd.getOptionValue(c));
        }
        return n;
    }

    public String getOptionValue(char c, String string) {
        if (this.cmd.hasOption(c)) {
            return this.cmd.getOptionValue(c);
        }
        return string;
    }
}

