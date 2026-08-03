# Keep the JNI bridge: native method registration is done by name at runtime.
-keep class com.ledgerlens.app.LlamaBridge { *; }

# kotlinx-serialization (only relevant if minification is ever enabled)
-keepattributes *Annotation*, InnerClasses
-keepclassmembers class com.ledgerlens.app.** {
    *** Companion;
}
-keepclasseswithmembers class com.ledgerlens.app.** {
    kotlinx.serialization.KSerializer serializer(...);
}
