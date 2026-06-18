package tech.kayys.tafkir.ml;

import org.junit.jupiter.api.Test;
import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.ml.nn.Parameter;
import tech.kayys.tafkir.ml.optim.Adam;
import tech.kayys.tafkir.ml.optim.AdamW;
import tech.kayys.tafkir.ml.optim.Adadelta;
import tech.kayys.tafkir.ml.optim.Adagrad;
import tech.kayys.tafkir.ml.optim.LAMB;
import tech.kayys.tafkir.ml.optim.Lion;
import tech.kayys.tafkir.ml.optim.Lookahead;
import tech.kayys.tafkir.ml.optim.NAdam;
import tech.kayys.tafkir.ml.optim.Optimizer;
import tech.kayys.tafkir.ml.optim.RAdam;
import tech.kayys.tafkir.ml.optim.RMSprop;
import tech.kayys.tafkir.ml.optim.SAM;
import tech.kayys.tafkir.ml.optim.SGD;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AljabrDLOptimizerFactoryTest {

    @Test
    void exposesCoreOptimizerFactoriesOnAljabrFacade() {
        assertInstanceOf(SGD.class, Aljabr.DL.sgd(parameters(), 0.1f));
        assertInstanceOf(SGD.class, Aljabr.DL.sgd(parameters(), 0.1f, 0.9f));
        assertInstanceOf(SGD.class, Aljabr.DL.sgd(parameters(), 0.1f, 0.9f, 0.01f, true));
    }

    @Test
    void exposesAdaptiveOptimizerFactoriesOnAljabrFacade() {
        assertInstanceOf(Adam.class, Aljabr.DL.adam(parameters(), 0.001f));
        assertInstanceOf(Adam.class, Aljabr.DL.adam(parameters(), 0.001f, 0.01f));
        assertInstanceOf(Adam.class, Aljabr.DL.adam(parameters(), 0.001f, 0.8f, 0.99f, 1e-6f, 0.01f, true));
        assertInstanceOf(AdamW.class, Aljabr.DL.adamW(parameters(), 0.001f));
        assertInstanceOf(AdamW.class, Aljabr.DL.adamW(parameters(), 0.001f, 0.05f));
        assertInstanceOf(AdamW.class, Aljabr.DL.adamW(parameters(), 0.001f, 0.8f, 0.99f, 1e-6f, 0.05f, true));
        assertInstanceOf(RMSprop.class, Aljabr.DL.rmsprop(parameters(), 0.01f));
        assertInstanceOf(RMSprop.class, Aljabr.DL.rmsprop(parameters(), 0.01f, 0.95f, 1e-6f, 0.01f, 0.9f));
        assertInstanceOf(NAdam.class, Aljabr.DL.nadam(parameters(), 0.001f));
        assertInstanceOf(NAdam.class, Aljabr.DL.nadam(parameters(), 0.001f, 0.01f));
        assertInstanceOf(NAdam.class, Aljabr.DL.nadamW(parameters(), 0.001f, 0.01f));
    }

    @Test
    void exposesAdvancedOptimizerFactoriesOnAljabrFacade() {
        assertInstanceOf(Adagrad.class, Aljabr.DL.adagrad(parameters(), 0.01f));
        assertInstanceOf(Adagrad.class, Aljabr.DL.adagrad(parameters(), 0.01f, 1e-8f, 0.01f));
        assertInstanceOf(Adadelta.class, Aljabr.DL.adadelta(parameters()));
        assertInstanceOf(Adadelta.class, Aljabr.DL.adadelta(parameters(), 1.0f, 0.95f, 1e-6f));
        assertInstanceOf(LAMB.class, Aljabr.DL.lamb(parameters(), 0.001f));
        assertInstanceOf(LAMB.class, Aljabr.DL.lamb(parameters(), 0.001f, 0.9f, 0.999f, 1e-6f, 0.01f));
        assertInstanceOf(Lion.class, Aljabr.DL.lion(parameters(), 0.0001f));
        assertInstanceOf(Lion.class, Aljabr.DL.lion(parameters(), 0.0001f, 0.9f, 0.99f, 0.01f));
        assertInstanceOf(RAdam.class, Aljabr.DL.radam(parameters(), 0.001f));
        assertInstanceOf(RAdam.class, Aljabr.DL.radam(parameters(), 0.001f, 0.01f));

        List<Parameter> lookaheadParams = parameters();
        Optimizer lookaheadBase = Aljabr.DL.sgd(lookaheadParams, 0.1f);
        assertInstanceOf(Lookahead.class, Aljabr.DL.lookahead(lookaheadBase));
        assertInstanceOf(Lookahead.class, Aljabr.DL.lookahead(Aljabr.DL.sgd(parameters(), 0.1f), 2, 0.6f));

        List<Parameter> samParams = parameters();
        Optimizer samBase = Aljabr.DL.sgd(samParams, 0.1f);
        assertInstanceOf(SAM.class, Aljabr.DL.sam(samParams, samBase));
        assertInstanceOf(SAM.class, Aljabr.DL.sam(samParams, samBase, 0.05f));
    }

    @Test
    @SuppressWarnings("removal")
    void deprecatedAljabrMlBridgeDelegatesToCoreOptimizerFactories() {
        assertInstanceOf(SGD.class, AljabrML.DL.sgd(parameters(), 0.1f, 0.8f));
        assertInstanceOf(Adam.class, AljabrML.DL.adam(parameters(), 0.001f, 0.01f));
        assertInstanceOf(AdamW.class, AljabrML.DL.adamW(parameters(), 0.001f, 0.02f));
        assertInstanceOf(RMSprop.class, AljabrML.DL.rmsprop(parameters(), 0.01f, 0.95f, 1e-6f, 0.01f, 0.8f));
    }

    @Test
    void factoryOptimizersCanStepAndExposeState() {
        Parameter parameter = parameter(1f, -2f);
        Optimizer optimizer = Aljabr.DL.rmsprop(List.of(parameter), 0.01f, 0.95f, 1e-6f, 0.01f, 0.9f);

        parameter.data().backward(GradTensor.of(new float[] {0.5f, -0.25f}, 2));
        optimizer.step();

        assertNotEquals(1f, parameter.data().data()[0], 1e-6f);
        assertNotEquals(-2f, parameter.data().data()[1], 1e-6f);
        assertTrue(optimizer.supportsStateDict());

        Map<String, Object> state = optimizer.stateDict();
        assertEquals("RMSprop", state.get("optimizer"));
        assertEquals(0.95f, ((Number) state.get("alpha")).floatValue(), 1e-7f);
        assertEquals(0.01f, ((Number) state.get("weightDecay")).floatValue(), 1e-7f);
        assertEquals(0.9f, ((Number) state.get("momentum")).floatValue(), 1e-7f);
    }

    private static List<Parameter> parameters() {
        return List.of(parameter(1f, -2f));
    }

    private static Parameter parameter(float... values) {
        return new Parameter(GradTensor.of(values, values.length));
    }
}
