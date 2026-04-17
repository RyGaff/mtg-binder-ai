package expo.modules.carddetector

import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class CardDetectorModule : Module() {

    companion object {
        init {
            System.loadLibrary("card_detector")
            // Register the frame processor plugin with VisionCamera's auto-linking registry
            FrameProcessorPluginRegistry.addFrameProcessorPlugin("detectCardCornersInFrame") { proxy, options ->
                CardDetectorFrameProcessorPlugin(proxy, options)
            }
        }
    }

    private external fun detectCornersNative(filePath: String): FloatArray?

    override fun definition() = ModuleDefinition {
        Name("CardDetector")

        AsyncFunction("detectCardCorners") { uri: String ->
            val path = if (uri.startsWith("file://")) uri.removePrefix("file://") else uri
            val raw = detectCornersNative(path) ?: return@AsyncFunction null
            if (raw.size != 8) return@AsyncFunction null
            mapOf(
                "topLeftX"     to raw[0], "topLeftY"     to raw[1],
                "topRightX"    to raw[2], "topRightY"    to raw[3],
                "bottomRightX" to raw[4], "bottomRightY" to raw[5],
                "bottomLeftX"  to raw[6], "bottomLeftY"  to raw[7],
            )
        }
    }
}
