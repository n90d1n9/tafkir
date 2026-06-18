package tech.kayys.tafkir.ml.hub;

/**
 * Exception thrown when model hub operations fail.
 */
public class HubException extends RuntimeException {

    public HubException(String message) {
        super(message);
    }

    public HubException(String message, Throwable cause) {
        super(message, cause);
    }
}
