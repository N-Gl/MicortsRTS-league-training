/*
 * Decompiled with CFR 0.152.
 */
package org.apache.commons.cli;

import java.util.Iterator;
import java.util.List;
import org.apache.commons.cli.ParseException;

public class MissingOptionException
extends ParseException {
    private static final long serialVersionUID = 8161889051578563249L;
    private List missingOptions;

    private static String createMessage(List<?> missingOptions) {
        StringBuilder buf = new StringBuilder("Missing required option");
        buf.append(missingOptions.size() == 1 ? "" : "s");
        buf.append(": ");
        Iterator<?> it = missingOptions.iterator();
        while (it.hasNext()) {
            buf.append(it.next());
            if (!it.hasNext()) continue;
            buf.append(", ");
        }
        return buf.toString();
    }

    public MissingOptionException(List missingOptions) {
        this(MissingOptionException.createMessage(missingOptions));
        this.missingOptions = missingOptions;
    }

    public MissingOptionException(String message) {
        super(message);
    }

    public List getMissingOptions() {
        return this.missingOptions;
    }
}

