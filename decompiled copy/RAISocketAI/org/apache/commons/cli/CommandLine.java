/*
 * Decompiled with CFR 0.152.
 */
package org.apache.commons.cli;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.TypeHandler;
import org.apache.commons.cli.Util;

public class CommandLine
implements Serializable {
    private static final long serialVersionUID = 1L;
    private final List<String> args = new LinkedList<String>();
    private final List<Option> options = new ArrayList<Option>();

    protected CommandLine() {
    }

    protected void addArg(String arg) {
        this.args.add(arg);
    }

    protected void addOption(Option opt) {
        this.options.add(opt);
    }

    public List<String> getArgList() {
        return this.args;
    }

    public String[] getArgs() {
        String[] answer = new String[this.args.size()];
        this.args.toArray(answer);
        return answer;
    }

    @Deprecated
    public Object getOptionObject(char opt) {
        return this.getOptionObject(String.valueOf(opt));
    }

    @Deprecated
    public Object getOptionObject(String opt) {
        try {
            return this.getParsedOptionValue(opt);
        }
        catch (ParseException pe) {
            System.err.println("Exception found converting " + opt + " to desired type: " + pe.getMessage());
            return null;
        }
    }

    public Properties getOptionProperties(Option option) {
        Properties props = new Properties();
        for (Option processedOption : this.options) {
            if (!processedOption.equals(option)) continue;
            List<String> values = processedOption.getValuesList();
            if (values.size() >= 2) {
                props.put(values.get(0), values.get(1));
                continue;
            }
            if (values.size() != 1) continue;
            props.put(values.get(0), "true");
        }
        return props;
    }

    public Properties getOptionProperties(String opt) {
        Properties props = new Properties();
        for (Option option : this.options) {
            if (!opt.equals(option.getOpt()) && !opt.equals(option.getLongOpt())) continue;
            List<String> values = option.getValuesList();
            if (values.size() >= 2) {
                props.put(values.get(0), values.get(1));
                continue;
            }
            if (values.size() != 1) continue;
            props.put(values.get(0), "true");
        }
        return props;
    }

    public Option[] getOptions() {
        List<Option> processed = this.options;
        Option[] optionsArray = new Option[processed.size()];
        return processed.toArray(optionsArray);
    }

    public String getOptionValue(char opt) {
        return this.getOptionValue(String.valueOf(opt));
    }

    public String getOptionValue(char opt, String defaultValue) {
        return this.getOptionValue(String.valueOf(opt), defaultValue);
    }

    public String getOptionValue(Option option) {
        if (option == null) {
            return null;
        }
        String[] values = this.getOptionValues(option);
        return values == null ? null : values[0];
    }

    public String getOptionValue(Option option, String defaultValue) {
        String answer = this.getOptionValue(option);
        return answer != null ? answer : defaultValue;
    }

    public String getOptionValue(String opt) {
        return this.getOptionValue(this.resolveOption(opt));
    }

    public String getOptionValue(String opt, String defaultValue) {
        return this.getOptionValue(this.resolveOption(opt), defaultValue);
    }

    public String[] getOptionValues(char opt) {
        return this.getOptionValues(String.valueOf(opt));
    }

    public String[] getOptionValues(Option option) {
        ArrayList<String> values = new ArrayList<String>();
        for (Option processedOption : this.options) {
            if (!processedOption.equals(option)) continue;
            values.addAll(processedOption.getValuesList());
        }
        return values.isEmpty() ? null : values.toArray(new String[values.size()]);
    }

    public String[] getOptionValues(String opt) {
        return this.getOptionValues(this.resolveOption(opt));
    }

    public Object getParsedOptionValue(char opt) throws ParseException {
        return this.getParsedOptionValue(String.valueOf(opt));
    }

    public Object getParsedOptionValue(Option option) throws ParseException {
        if (option == null) {
            return null;
        }
        String res = this.getOptionValue(option);
        if (res == null) {
            return null;
        }
        return TypeHandler.createValue(res, option.getType());
    }

    public Object getParsedOptionValue(String opt) throws ParseException {
        return this.getParsedOptionValue(this.resolveOption(opt));
    }

    public boolean hasOption(char opt) {
        return this.hasOption(String.valueOf(opt));
    }

    public boolean hasOption(Option opt) {
        return this.options.contains(opt);
    }

    public boolean hasOption(String opt) {
        return this.hasOption(this.resolveOption(opt));
    }

    public Iterator<Option> iterator() {
        return this.options.iterator();
    }

    private Option resolveOption(String opt) {
        opt = Util.stripLeadingHyphens(opt);
        for (Option option : this.options) {
            if (!opt.equals(option.getOpt()) && !opt.equals(option.getLongOpt())) continue;
            return option;
        }
        return null;
    }

    public static final class Builder {
        private final CommandLine commandLine = new CommandLine();

        public Builder addArg(String arg) {
            this.commandLine.addArg(arg);
            return this;
        }

        public Builder addOption(Option opt) {
            this.commandLine.addOption(opt);
            return this;
        }

        public CommandLine build() {
            return this.commandLine;
        }
    }
}

