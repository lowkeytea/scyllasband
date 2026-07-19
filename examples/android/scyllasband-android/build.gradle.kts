plugins {
    alias(libs.plugins.android.library)
}

val onnxRuntimeVersion = "1.23.2"
val onnxRuntimeAar by configurations.creating
val libscyllasbandDir = projectDir.resolve("../../../libscyllasband").canonicalFile
val extractedOnnxRuntimeDir = layout.buildDirectory.dir("intermediates/onnxruntime/android")
val onnxRuntimeArtifacts = onnxRuntimeAar.incoming.artifactView {}.files

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
                    "-DSCYLLASBAND_ENABLE_ONNX=ON",
                    "-DSCYLLASBAND_BUILD_TOOLS=OFF",
                    "-DSCYLLASBAND_BUILD_TESTS=OFF",
                    "-DSCYLLASBAND_ONNXRUNTIME_INCLUDE_DIR=${extractedOnnxRuntimeDir.get().asFile.absolutePath}/headers",
                    "-DSCYLLASBAND_ANDROID_ONNXRUNTIME_ROOT=${extractedOnnxRuntimeDir.get().asFile.absolutePath}",
                )
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
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:$onnxRuntimeVersion")
    onnxRuntimeAar("com.microsoft.onnxruntime:onnxruntime-android:$onnxRuntimeVersion@aar")
}

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
