plugins {
    alias(libs.plugins.android.application)
}

val repositoryRoot = projectDir.resolve("../../..").canonicalFile
val scyllasbandOnnxInt8BundleDir = repositoryRoot.resolve("scyllasband/models/onnx-int8")
val exampleDataDir = repositoryRoot.resolve("data")
val generatedAssetsDir = layout.buildDirectory.dir("generated/scyllasbandAssets/main")

val prepareScyllasBandAssets by tasks.registering(Sync::class) {
    doFirst {
        check(scyllasbandOnnxInt8BundleDir.resolve("manifest.json").isFile) {
            "Scylla's Band ONNX manifest not found at ${scyllasbandOnnxInt8BundleDir.absolutePath}. Run `python -m scyllasband download --runtime-bundles onnx-int8`."
        }
        check(scyllasbandOnnxInt8BundleDir.resolve("onnx/components/shared_weights.bin").isFile) {
            "Scylla's Band ONNX shared weights are missing under ${scyllasbandOnnxInt8BundleDir.absolutePath}."
        }
        check(scyllasbandOnnxInt8BundleDir.resolve("onnx/g2p/model.onnx").isFile) {
            "Scylla's Band ONNX G2P model is missing under ${scyllasbandOnnxInt8BundleDir.absolutePath}."
        }
    }
    from(scyllasbandOnnxInt8BundleDir) {
        into("scyllasband/onnx-int8")
    }
    from(exampleDataDir) {
        include("emotional_text.txt", "groupSpeak.txt", "test_document.txt")
        into("scyllasband/examples")
    }
    into(generatedAssetsDir)
}

android {
    namespace = "org.scyllasband.demo"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "org.scyllasband.demo"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets.named("main") {
        assets.directories.add(generatedAssetsDir.get().asFile.absolutePath)
    }

    androidResources {
        noCompress += listOf("onnx", "bin", "npz")
    }
}

tasks.configureEach {
    if (
        name != "prepareScyllasBandAssets" &&
        (name.contains("Assets") || name.contains("lint", ignoreCase = true))
    ) {
        dependsOn(prepareScyllasBandAssets)
    }
}

dependencies {
    implementation(project(":scyllasband-android"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
