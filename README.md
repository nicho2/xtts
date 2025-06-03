# xtts




pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121


## To install deepspeed on Windows

  - follow : https://github.com/erew123/deepspeedpatcher (useful tool)
    - in my case, i have commented :


      #ifndef BF16_AVAILABLE
        //using __nv_bfloat162 = __half2;
      #endif
  in gelu.cu and transform.cu

   https://github.com/microsoft/DeepSpeed/issues/6709
   
 - and pip install deepspeed_wheels/deepspeed_0.16.2_cuda12.6_py310/deepspeed-0.16.2+unknown-cp310-cp310-win_amd64.whl

