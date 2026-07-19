package org.scyllasband.android

import android.content.Context
import java.io.File

internal class ScyllasBandAssetInstaller(
    private val context: Context,
) {
    fun install(assetRoot: String = DEFAULT_ASSET_ROOT): File {
        val destination = File(context.noBackupFilesDir, assetRoot)
        val bundledManifest = context.assets.open("$assetRoot/manifest.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        val installedManifest = File(destination, "manifest.json")
        val installedWeights = File(destination, "onnx/components/shared_weights.bin")
        if (
            installedManifest.isFile &&
            installedWeights.isFile &&
            installedManifest.readText(Charsets.UTF_8) == bundledManifest
        ) {
            return destination
        }

        val temporary = File(destination.parentFile, "${destination.name}.installing")
        temporary.deleteRecursively()
        temporary.mkdirs()
        copyAssetTree(assetRoot, temporary)
        check(File(temporary, "manifest.json").isFile) { "Installed Scylla's Band manifest is missing" }
        check(File(temporary, "onnx/components/shared_weights.bin").isFile) {
            "Installed Scylla's Band shared weights are missing"
        }
        destination.deleteRecursively()
        check(temporary.renameTo(destination)) {
            "Unable to move the installed Scylla's Band bundle to ${destination.absolutePath}"
        }
        return destination
    }

    private fun copyAssetTree(assetPath: String, destination: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            destination.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                destination.outputStream().buffered().use { output -> input.copyTo(output) }
            }
            return
        }
        destination.mkdirs()
        children.forEach { child ->
            copyAssetTree("$assetPath/$child", File(destination, child))
        }
    }

    private companion object {
        const val DEFAULT_ASSET_ROOT = "scyllasband/onnx"
    }
}
