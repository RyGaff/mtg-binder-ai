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
            val raw = detectCornersNative(path)
                ?: return@AsyncFunction mapOf<String, Any>("_error" to "detect_failed")
            if (raw.size != 10) {
                return@AsyncFunction mapOf<String, Any>("_error" to "detect_bad_shape")
            }

            val rectPath = "$path.rect.jpg"
            val rectFile = File(rectPath)
            // Matches CardSource enum in card_detector.h (0/1/2).
            val source = when (raw[9].toInt()) {
                1    -> "lineinterp"
                2    -> "otsu"
                else -> "primary"
            }

            mapOf(
                "topLeftX"     to raw[0], "topLeftY"     to raw[1],
                "topRightX"    to raw[2], "topRightY"    to raw[3],
                "bottomRightX" to raw[4], "bottomRightY" to raw[5],
                "bottomLeftX"  to raw[6], "bottomLeftY"  to raw[7],
                "confidence"   to raw[8],
                "source"       to source,
                "rectifiedUri" to if (rectFile.exists()) "file://$rectPath" else null,
            )
        }

        AsyncFunction("encodeImage") { uri: String ->
            val itp = loadInterpreter()
                ?: return@AsyncFunction mapOf<String, Any>("error" to "interpreter not loaded (card_encoder.tflite missing from assets?)")
            val path = if (uri.startsWith("file://")) uri.removePrefix("file://") else uri
            val bmp = BitmapFactory.decodeFile(path)
                ?: return@AsyncFunction mapOf<String, Any>("error" to "image decode failed: $path")

            // Mirror albumentations training transform:
            //   LongestMaxSize(224) + PadIfNeeded(224, 224, fill=0)
            // so card aspect ratio is preserved and the short side is
            // center-padded with black, matching what the encoder was trained
            // on. Otherwise a card (aspect ≈ 0.71) gets squashed to square.
            val side = 224
            val scale = side.toFloat() / maxOf(bmp.width, bmp.height)
            val drawW = (bmp.width * scale).toInt().coerceAtLeast(1)
            val drawH = (bmp.height * scale).toInt().coerceAtLeast(1)
            val resized = Bitmap.createScaledBitmap(bmp, drawW, drawH, true)
            bmp.recycle()

            val canvas = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
            val canvasPainter = android.graphics.Canvas(canvas)
            canvasPainter.drawColor(android.graphics.Color.BLACK)
            val offsetX = (side - drawW) / 2f
            val offsetY = (side - drawH) / 2f
            canvasPainter.drawBitmap(resized, offsetX, offsetY, null)
            resized.recycle()

            val input = ByteBuffer.allocateDirect(4 * side * side * 3)
                .order(ByteOrder.nativeOrder())
            val pixels = IntArray(side * side)
            canvas.getPixels(pixels, 0, side, 0, 0, side, side)
            canvas.recycle()

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
                return@AsyncFunction mapOf<String, Any>(
                    "error" to "inference failed: ${e.message ?: e.javaClass.simpleName}"
                )
            }
            return@AsyncFunction mapOf<String, Any>(
                "embedding" to output[0].toList().map { it.toDouble() }
            )
        }
    }
}
