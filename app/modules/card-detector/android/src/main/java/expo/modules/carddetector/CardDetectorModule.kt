package expo.modules.carddetector

import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.File

class CardDetectorModule : Module() {

    companion object {
        init {
            System.loadLibrary("card_detector")
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
            if (raw.size != 9) return@AsyncFunction null

            val rectPath = "$path.rect.jpg"
            val rectFile = File(rectPath)

            mapOf(
                "topLeftX"     to raw[0], "topLeftY"     to raw[1],
                "topRightX"    to raw[2], "topRightY"    to raw[3],
                "bottomRightX" to raw[4], "bottomRightY" to raw[5],
                "bottomLeftX"  to raw[6], "bottomLeftY"  to raw[7],
                "confidence"   to raw[8],
                "rectifiedUri" to if (rectFile.exists()) "file://$rectPath" else null,
            )
        }
    }
}
