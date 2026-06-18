package tech.kayys.tafkir.jupyter;

import org.dflib.jjava.jupyter.channels.JupyterConnection;
import org.dflib.jjava.jupyter.kernel.KernelConnectionProperties;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Entry point registered in {@code kernel.json} argv.
 * Jupyter passes the connection file path as the sole argument.
 */
public class KernelLauncher {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: KernelLauncher <connection-file>");
            System.exit(1);
        }

        String json = Files.readString(Paths.get(args[0]));
        KernelConnectionProperties props = KernelConnectionProperties.parse(json);

        TafkirKernel kernel = new TafkirKernel();
        JupyterConnection connection = new JupyterConnection(props);

        kernel.becomeHandlerForConnection(connection);
        connection.connect();
        connection.waitUntilClose();
    }
}
