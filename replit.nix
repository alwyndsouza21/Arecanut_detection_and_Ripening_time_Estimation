{ pkgs }: {
  deps = [
    pkgs.zlib
    pkgs.which
    pkgs.snappy
    pkgs.openssl
    pkgs.nsync
    pkgs.libjpeg_turbo
    pkgs.jsoncpp
    pkgs.grpc
    pkgs.gitFull
    pkgs.giflib
    pkgs.double-conversion
    pkgs.curl
    pkgs.bazel
    pkgs.glibcLocales
    pkgs.glibc
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
    pkgs.python310Full
    pkgs.replitPackages.prybar-python310
    pkgs.replitPackages.stderred
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.glibcLocales
      pkgs.tk
      pkgs.tcl
      pkgs.qhull
      pkgs.gtk3
      pkgs.gobject-introspection
      pkgs.ghostscript
      pkgs.freetype
      pkgs.cairo
      # Needed for pandas / numpy
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      # Needed for pygame
      pkgs.glib
      # Needed for matplotlib
      pkgs.xorg.libX11
      # Needed for tensorflow with GPU
      pkgs.cudatoolkit
      pkgs.cudaPackages.cuda_cudart
      pkgs.cudaPackages.cudnn
      pkgs.cudaPackages.libcublas
      pkgs.cudaPackages.libcufft
      pkgs.cudaPackages.libcurand
      pkgs.cudaPackages.libcusolver
      pkgs.cudaPackages.libcusparse
    ];
    PYTHONHOME = "${pkgs.python310Full}";
    PYTHONBIN = "${pkgs.python310Full}/bin/python3.10";
    LANG = "en_US.UTF-8";
    STDERREDBIN = "${pkgs.replitPackages.stderred}/bin/stderred";
    PRYBAR_PYTHON_BIN = "${pkgs.replitPackages.prybar-python310}/bin/prybar-python310";
  };
}