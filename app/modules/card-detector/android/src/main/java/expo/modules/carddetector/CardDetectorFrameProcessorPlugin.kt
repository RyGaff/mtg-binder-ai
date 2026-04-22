package expo.modules.carddetector

import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy

class CardDetectorFrameProcessorPlugin(
    proxy: VisionCameraProxy,
    options: Map<String, Any>?
) : FrameProcessorPlugin(proxy, options) {

    companion object {
        init {
            System.loadLibrary("card_detector")
        }
    }

    private external fun detectCornersFromGrayscaleNative(
        bytes: ByteArray, width: Int, height: Int
    ): FloatArray?

    override fun callback(frame: Frame, arguments: Map<String, Any>?): Any? {
        val image = frame.image
        val plane = image.planes[0]
        val buffer = plane.buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        val raw = detectCornersFromGrayscaleNative(bytes, image.width, image.height)
            ?: return null
        if (raw.size != 10) return null
        val source = when (raw[9].toInt()) {
            1    -> "lineinterp"
            2    -> "otsu"
            else -> "primary"
        }

        return mapOf(
            "topLeftX"     to raw[0].toDouble(), "topLeftY"     to raw[1].toDouble(),
            "topRightX"    to raw[2].toDouble(), "topRightY"    to raw[3].toDouble(),
            "bottomRightX" to raw[4].toDouble(), "bottomRightY" to raw[5].toDouble(),
            "bottomLeftX"  to raw[6].toDouble(), "bottomLeftY"  to raw[7].toDouble(),
            "confidence"   to raw[8].toDouble(),
            "source"       to source,
            // Frame processor path does not produce a rectified image
        )
    }
}
