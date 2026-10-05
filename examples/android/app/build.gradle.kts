plugins {
    alias(libs.plugins.android.application)
}

// LiteRT (default) or ONNX Runtime: -Pscyllasband.backend=onnx. The library module reads the same property.
val scyllasbandBackend = (findProperty("scyllasband.backend") as String?)?.lowercase() ?: "litert"
val repositoryRoot = projectDir.resolve("../../..").canonicalFile
val scyllasbandBundleDir = repositoryRoot.resolve("scyllasband/models/$scyllasbandBackend")
val exampleDataDir = repositoryRoot.resolve("data")
val generatedAssetsDir = layout.buildDirectory.dir("generated/scyllasbandAssets/main")

val prepareScyllasBandAssets by tasks.registering(Sync::class) {
    doFirst {
        check(scyllasbandBundleDir.resolve("manifest.json").isFile) {
            "Scylla's Band $scyllasbandBackend bundle not found at ${scyllasbandBundleDir.absolutePath}. " +
                "Run `python -m scyllasband download --yes`" +
                (if (scyllasbandBackend == "onnx") " --flavor onnx" else "") + "."
        }
    }
    from(scyllasbandBundleDir) {
        into("scyllasband/bundle")
    }
    from(exampleDataDir) {
        include("walkthrough_demo.txt", "test_document.txt", "emotional_text.txt")
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
        noCompress += listOf("tflite", "onnx", "bin")
    }

    packaging {
        // LiteRT loads its GPU accelerator from the directory of libLiteRt.so, so the libraries must be
        // extracted to disk instead of being mapped from the APK.
        jniLibs.useLegacyPackaging = true
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
