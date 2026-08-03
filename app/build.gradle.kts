plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
    alias(libs.plugins.protobuf)
    alias(libs.plugins.hilt)
    alias(libs.plugins.kotlin.android)
}

// fork-specific: conditional release signing via keystore.properties
// Upstream intentionally omits signingConfigs for reproducible builds.
// This block is a no-op when keystore.properties is absent.
import java.io.FileInputStream
import java.util.Properties
import java.net.URI
import java.security.MessageDigest

val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = Properties()
if (keystorePropertiesFile.exists()) {
    keystoreProperties.load(FileInputStream(keystorePropertiesFile))
}

android {
    namespace = "dev.wceng.sufei"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.wceng.sufei"
        minSdk = 24
        targetSdk = 35
        versionCode = 10
        versionName = "1.6.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        if (keystorePropertiesFile.exists()) {
            create("release") {
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
                storeFile = file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
            }
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            if (keystorePropertiesFile.exists()) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }

    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }

    sourceSets {
        getByName("androidTest") {
            assets.srcDirs(files("$projectDir/schemas"))
        }
    }
}


ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}


kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.core.splashscreen)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.kotlinx.serialization.json)

    // Room
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    // DataStore & Protobuf
    implementation(libs.androidx.datastore)
    implementation(libs.protobuf.kotlin.lite)

    // Hilt
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)
    implementation(libs.androidx.hilt.navigation.compose)

    // Navigation 3
    implementation(libs.androidx.navigation3.runtime)
    implementation(libs.androidx.navigation3.ui)
    implementation(libs.androidx.lifecycle.viewModel.navigation3)

    // OpenCC4J
    implementation(libs.opencc4j)

    // Adaptive Navigation
    implementation(libs.androidx.compose.material3.adaptive.navigation.suite)

    // Glance AppWidget
    implementation(libs.glance.appwidget)
    implementation(libs.glance.material3)

    testImplementation(libs.junit)
    testImplementation(libs.mockk)
    testImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.room.testing)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    // MOSS-TTS-Nano ONNX Runtime (exploration, issue #37)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.27.0")
    testImplementation("org.json:json:20240303")
}

protobuf {
    protoc {
        artifact = libs.protobuf.protoc.get().toString()
    }
    generateProtoTasks {
        all().forEach { task ->
            task.builtins {
                register("java") {
                    option("lite")
                }
                register("kotlin") {
                    option("lite")
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fetchTtsModels — pull & verify the on-device NAR TTS model (m3_v6).
//
// The ~50 MB fp16 model is NOT tracked in git (see .gitignore). It is fetched
// from the Hugging Face model repo at a pinned commit, each file verified
// against its known sha256, then written to assets/models/nar/ so the APK
// bundles it. NarModelAssets extracts it to filesDir on first launch, keyed on
// the generated SHA256SUMS.txt marker.
//
// Model version bump  = change ttsModelRevision + the sha256 map below, ship a
//                       new APK (no runtime OTA).
// China networks      = -PhfEndpoint=https://hf-mirror.com  (or env HF_ENDPOINT)
// Build without model = -PembedTtsModel=false
// ---------------------------------------------------------------------------
val ttsModelRepo = "Sopho/sopho-poetry-tts"
// Pinned m3v6 deploy commit on huggingface.co/Sopho/sopho-poetry-tts (tag v0.1).
val ttsModelRevision = "baf26a5416a47d08c04f4d7378726ff39c84037f"
val ttsModelSha256 = linkedMapOf(
    "fastspeech2_sufei.onnx" to "9e58042c158a28aa8391586a42f4f83f2abc5af6640851c9414dbf731d378d9c",
    "hifigan_csmsc.onnx"     to "eb305fdc8be9412d752063d01926cd631c9816a3bdc73fd5a831e0c131500c5f",
    "phone_id_map.txt"       to "ca00d60618a00a2dc1e42002faaa9024c9e5057e04b1e834325950111cee59c3",
    "norm_stats.npz"         to "19304d7074d29a298c084d8e5df0097b3945156412e5c4443c9956ce4fb0cb15",
)

fun sha256Of(f: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    f.inputStream().use { s ->
        val buf = ByteArray(1 shl 16)
        while (true) {
            val n = s.read(buf)
            if (n < 0) break
            md.update(buf, 0, n)
        }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
}

fun downloadTo(url: String, dst: File) {
    var current = url
    for (redirect in 0..5) {
        val conn = URI(current).toURL().openConnection() as java.net.HttpURLConnection
        conn.instanceFollowRedirects = false
        conn.connectTimeout = 30_000
        conn.readTimeout = 180_000
        conn.setRequestProperty("User-Agent", "sufei-gradle-fetchTtsModels")
        when (val code = conn.responseCode) {
            301, 302, 303, 307, 308 -> {
                // Location may be relative per RFC 7231 — resolve it against the current URL.
                current = URI(current).resolve(conn.getHeaderField("Location")).toString()
                conn.disconnect()
            }
            200 -> {
                val tmp = File(dst.parentFile, "${dst.name}.part")
                conn.inputStream.use { input -> tmp.outputStream().use { input.copyTo(it) } }
                conn.disconnect()
                if (dst.exists()) dst.delete()
                if (!tmp.renameTo(dst)) { tmp.copyTo(dst, overwrite = true); tmp.delete() }
                return
            }
            else -> { conn.disconnect(); error("HTTP $code fetching $url") }
        }
    }
    error("too many redirects fetching $url")
}

val fetchTtsModels by tasks.registering {
    group = "tts"
    description = "Download & sha256-verify the m3_v6 NAR TTS model into assets/models/nar/."

    val embed = (project.findProperty("embedTtsModel") as String?)?.toBoolean() ?: true
    val endpoint = (project.findProperty("hfEndpoint") as String?)
        ?: System.getenv("HF_ENDPOINT") ?: "https://huggingface.co"
    val outDir = layout.projectDirectory.dir("src/main/assets/models/nar").asFile

    onlyIf { embed }
    inputs.property("revision", ttsModelRevision)
    inputs.property("sha256", ttsModelSha256)
    inputs.property("endpoint", endpoint)
    outputs.dir(outDir)

    doLast {
        check(!ttsModelRevision.startsWith("__")) {
            "ttsModelRevision is a placeholder — pin the real HF commit SHA before building with the embedded model."
        }
        outDir.mkdirs()
        val sums = StringBuilder()
        for ((name, sha) in ttsModelSha256) {
            val dst = File(outDir, name)
            if (dst.isFile && sha256Of(dst) == sha) {
                logger.lifecycle("fetchTtsModels: $name up-to-date")
            } else {
                val url = "$endpoint/$ttsModelRepo/resolve/$ttsModelRevision/$name?download=true"
                logger.lifecycle("fetchTtsModels: downloading $name from $url")
                downloadTo(url, dst)
                val got = sha256Of(dst)
                check(got == sha) { "sha256 mismatch for $name: expected $sha but got $got" }
            }
            sums.append("$sha  $name\n")
        }
        // The marker NarModelAssets reads to detect a shipped model-version change.
        File(outDir, "SHA256SUMS.txt").writeText(sums.toString())
        logger.lifecycle("fetchTtsModels: ${ttsModelSha256.size} files verified in $outDir")
    }
}

tasks.named("preBuild").configure { dependsOn(fetchTtsModels) }
