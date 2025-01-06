# Run Qualcomm SD via onnxruntime

## Download SD bin files from Qualcomm

https://aihub.qualcomm.com/compute/models/stable_diffusion_v2_1_quantized?searchTerm=stable

## Export json and generate onnx

```
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility --context_binary="$bin_file" --json_file="${bin_file%.bin}.json"

python gen_qnn_ctx_onnx_model.py -b "$bin_file" -q "${bin_file%.bin}.json" --quantized_IO --disable_embed_mode
```

For token encoder, set `--quantized_IO = False` because in bin file, int to float uses linear conversion but float to int uses a complex one, see `float_to_tfN_uint16`

## Prepare dlls

Copy `consts.QNN_SDK_ROOT + "\\lib\\arm64x-windows-msvc"` and `onsts.QNN_SDK_ROOT + "\\lib\\hexagon-v{}\\unsigned".format(consts.DSP_ARCH)`

DSP_ARCH = "73"  # For X-Elite device.

See https://github.com/quic/wos-ai-plugins/blob/main/plugins/stable-diffusion-webui/qairt_accelerate/install.py

## (optional) Build onnxruntime with same version of QNN

Not sure why.

Checkout commit

```
commit eb9b377306c941b01d2823b7655f372a20b82197 (HEAD)
Author: Adrian Lizarraga <adlizarraga@microsoft.com>
Date:   Wed Jul 24 10:17:12 2024 -0700

    [QNN EP] Update to QNN SDK 2.24.0 (#21463)
```

Comment D:\onnxruntime\cmake\onnxruntime_python.cmake line 223:   target_link_libraries(onnxruntime_pybind11_state PRIVATE ${Python_LIBRARY_RELEASE})

Then use `.\build.bat --arm64ec --use_qnn --qnn_home D:\Downloads\2.24.0.240626 --build_wheel  --skip_submodule_sync --config Release --build_dir .\build\qnn-arm64ec --skip_tests --path_to_protoc_exe D:\onnxruntime\build\qnn\Release\_deps\protobuf-build\Release\protoc.exe`
