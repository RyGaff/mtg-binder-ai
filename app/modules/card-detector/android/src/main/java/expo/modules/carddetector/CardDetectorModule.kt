package expo.modules.carddetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

class CardDetectorModule : Module() {

    private var interpreter: Interpreter? = null
    private var interpreterLoaded = false

    /**
     * Lazily load the TFLite encoder from the Android assets directory.
     * Returns null (and stays null for the process lifetime) when the asset
     * is not bundled in this build — the consumer-side JS wrapper treats
     * that as "feature disabled". Safe to call many times.
     */
    private fun loadInterpreter(): Interpreter? {
        if (interpreterLoaded) return interpreter
        interpreterLoaded = true
        try {
            val ctx = appContext.reactContext ?: return null
            val fd = ctx.assets.openFd("card_encoder.tflite")
            // MappedByteBuffer persists beyond the FileInputStream close — the OS
            // mapping is independent of the Java stream. TFLite 2.14 copies the
            // model bytes during Interpreter construction, so the close is safe.
            FileInputStream(fd.fileDescriptor).use { stream ->
                val buf = stream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset, fd.declaredLength,
                )
                interpreter = Interpreter(buf)
            }
        } catch (e: Exception) {
            Log.w("CardDetector", "card_encoder.tflite not in assets — encodeImage disabled", e)
            interpreter = null
        }
        return interpreter
    }

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

        AsyncFunction("encodeImage") { uri: String ->
            val itp = loadInterpreter() ?: return@AsyncFunction null
            val path = if (uri.startsWith("file://")) uri.removePrefix("file://") else uri
            val bmp = BitmapFactory.decodeFile(path) ?: return@AsyncFunction null
            val resized = Bitmap.createScaledBitmap(bmp, 224, 224, true)
            bmp.recycle()

            val input = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
                .order(ByteOrder.nativeOrder())
            val pixels = IntArray(224 * 224)
            resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
            resized.recycle()

            for (p in pixels) {
                input.putFloat(((p shr 16) and 0xFF) / 255f)  // R
                input.putFloat(((p shr 8)  and 0xFF) / 255f)  // G
                input.putFloat((p and 0xFF)           / 255f)  // B
            }
            input.rewind()

            val output = Array(1) { FloatArray(256) }
            try {
                itp.run(input, output)
            } catch (e: Exception) {
                Log.w("CardDetector", "TFLite inference failed", e)
                return@AsyncFunction null
            }
            return@AsyncFunction output[0].toList().map { it.toDouble() }
        }
    }
}
