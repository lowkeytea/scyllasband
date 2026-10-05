plugins {
    alias(libs.plugins.android.library)
}

// LiteRT (default) or ONNX Runtime: -Pscyllasband.backend=onnx. The app module reads the same property.
val scyllasbandBackend = (findProperty("scyllasband.backend") as String?)?.lowercase() ?: "litert"
require(scyllasbandBackend == "litert" || scyllasbandBackend == "onnx") {
    "scyllasband.backend must be litert or onnx (got '$scyllasbandBackend')"
}

val onnxRuntimeVersion = "1.30.0"
val libscyllasbandDir = projectDir.resolve("../../../libscyllasband").canonicalFile
val litertLibDir = libscyllasbandDir.resolve("third_party/litert/lib")
val onnxRuntimeAar by configurations.creating
val extractedOnnxRuntimeDir = layout.buildDirectory.dir("intermediates/onnxruntime/android")
val litertJniLibsDir = layout.buildDirectory.dir("generated/litertJniLibs")

android {
    namespace = "org.scyllasband.android"
    ndkVersion = "29.0.13113456"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        minSdk = 30
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        consumerProguardFiles("consumer-rules.pro")

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DSCYLLASBAND_BACKEND=$scyllasbandBackend",
                    "-DSCYLLASBAND_BUILD_TOOLS=OFF",
                    "-DSCYLLASBAND_BUILD_TESTS=OFF",
                )
                if (scyllasbandBackend == "onnx") {
                    arguments += "-DSCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT=" +
                        extractedOnnxRuntimeDir.get().asFile.absolutePath
                }
            }
        }
    }

    buildTypes {
        release {
            consumerProguardFiles("consumer-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
        }
    }

    if (scyllasbandBackend == "litert") {
        sourceSets.named("main") {
            jniLibs.directories.add(litertJniLibsDir.get().asFile.absolutePath)
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    testImplementation(libs.junit)
    if (scyllasbandBackend == "onnx") {
        implementation("com.microsoft.onnxruntime:onnxruntime-android:$onnxRuntimeVersion")
        onnxRuntimeAar("com.microsoft.onnxruntime:onnxruntime-android:$onnxRuntimeVersion@aar")
    }
}

if (scyllasbandBackend == "litert") {
    // Packages the staged libLiteRt.so and its GPU accelerator under the Android ABI names. libLiteRt is
    // linked by the native library; the accelerator is dlopen'ed from libLiteRt's directory at runtime.
    val stageLitertJniLibs by tasks.registering(Sync::class) {
        doFirst {
            listOf("android-arm64", "android-x86_64").forEach { platform ->
                check(litertLibDir.resolve("$platform/libLiteRt.so").isFile) {
                    "libLiteRt.so is not staged for $platform. Run `python libscyllasband/scripts/stage_litert_sdk.py " +
                        "--platform $platform --download-runtime --download-gpu-accelerator --overwrite`."
                }
            }
        }
        from(litertLibDir.resolve("android-arm64")) {
            include("*.so")
            into("arm64-v8a")
        }
        from(litertLibDir.resolve("android-x86_64")) {
            include("*.so")
            into("x86_64")
        }
        into(litertJniLibsDir)
    }
    tasks.configureEach {
        if (name.startsWith("configureCMake") || name.contains("JniLibFolders") || name.contains("NativeLibs")) {
            dependsOn(stageLitertJniLibs)
        }
    }
} else {
    val onnxRuntimeArtifacts = onnxRuntimeAar.incoming.artifactView {}.files
    val extractOnnxRuntimeAar by tasks.registering(Sync::class) {
        from(
            onnxRuntimeArtifacts.elements.map { artifacts ->
                artifacts.map { artifact -> zipTree(artifact.asFile) }
            },
        )
        into(extractedOnnxRuntimeDir)
    }
    tasks.configureEach {
        if (name.startsWith("configureCMake")) {
            dependsOn(extractOnnxRuntimeAar)
        }
    }
}
