# PyInstaller runtime hook: point DeepFilterNet at the model bundled in the exe.
# df.utils.get_cache_dir() calls appdirs.user_cache_dir("DeepFilterNet"); when frozen and
# the bundled model exists, return the bundle dir so no download/network is needed.
import os, sys
if getattr(sys, "frozen", False):
    _bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "df_models")
    if os.path.isfile(os.path.join(_bundled, "DeepFilterNet3", "config.ini")):
        import appdirs
        _orig = appdirs.user_cache_dir
        def user_cache_dir(appname=None, *a, **k):
            if appname == "DeepFilterNet":
                return _bundled
            return _orig(appname, *a, **k)
        appdirs.user_cache_dir = user_cache_dir
