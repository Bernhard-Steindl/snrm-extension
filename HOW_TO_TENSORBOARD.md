# Run TensorBoard


https://github.com/tensorflow/tensorboard

https://www.tensorflow.org/tensorboard/get_started



    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % pwd
    /Users/bernhardsteindl/Development/python_workspace/bachelorarbeit/snrm-extension
    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % conda activate snrm-tf1
    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % conda install -c conda-forge tensorboard


    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % conda list --explicit
    # This file may be used to create an environment using:
    # $ conda create --name <env> --file <this file>
    # platform: osx-64
    @EXPLICIT
    https://conda.anaconda.org/conda-forge/osx-64/ca-certificates-2019.11.28-hecc5488_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/libcxxabi-4.0.1-hcfea43d_1.conda
    https://conda.anaconda.org/conda-forge/osx-64/llvm-openmp-9.0.0-h40edb58_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/xz-5.2.4-h1de35cc_4.conda
    https://repo.anaconda.com/pkgs/main/osx-64/zlib-1.2.11-h1de35cc_3.conda
    https://repo.anaconda.com/pkgs/main/osx-64/libcxx-4.0.1-hcfea43d_1.conda
    https://conda.anaconda.org/conda-forge/osx-64/libgfortran-4.0.0-2.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/openssl-1.1.1d-h0b31af3_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/tk-8.6.8-ha441bb4_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/libffi-3.2.1-h475c297_4.conda
    https://conda.anaconda.org/conda-forge/osx-64/libopenblas-0.3.7-h4bb4525_2.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/libprotobuf-3.9.2-hd9629dc_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/ncurses-6.1-h0a44026_1.conda
    https://conda.anaconda.org/conda-forge/osx-64/libblas-3.8.0-14_openblas.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/libedit-3.1.20181209-hb402a30_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/readline-7.0-h1de35cc_5.conda
    https://conda.anaconda.org/conda-forge/osx-64/libcblas-3.8.0-14_openblas.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/liblapack-3.8.0-14_openblas.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/sqlite-3.30.1-ha441bb4_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/python-3.6.9-h359304d_0.conda
    https://conda.anaconda.org/conda-forge/osx-64/certifi-2019.11.28-py36_0.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/numpy-1.16.4-py36h6b0580a_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/six-1.12.0-py36_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/webencodings-0.5.1-py36_1.conda
    https://repo.anaconda.com/pkgs/main/noarch/werkzeug-0.16.0-py_0.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/html5lib-0.9999999-py36_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/nltk-3.4.5-py36_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/setuptools-41.4.0-py36_0.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/bleach-1.5.0-py36_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/markdown-3.1.1-py36_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/protobuf-3.9.2-py36h0a44026_0.conda
    https://repo.anaconda.com/pkgs/main/osx-64/wheel-0.33.6-py36_0.tar.bz2
    https://repo.anaconda.com/pkgs/main/osx-64/pip-19.3.1-py36_0.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/tensorboard-0.4.0rc3-py36_2.tar.bz2
    https://conda.anaconda.org/conda-forge/osx-64/tensorflow-1.4.0-py36_0.tar.bz2



    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % conda list
    # packages in environment at /Users/bernhardsteindl/anaconda3/envs/snrm-tf1:
    #
    # Name                    Version                   Build  Channel
    bleach                    1.5.0                    py36_0    conda-forge
    ca-certificates           2019.11.28           hecc5488_0    conda-forge
    certifi                   2019.11.28               py36_0    conda-forge
    dill                      0.3.1.1                  pypi_0    pypi
    html5lib                  0.9999999                py36_0    conda-forge
    libblas                   3.8.0               14_openblas    conda-forge
    libcblas                  3.8.0               14_openblas    conda-forge
    libcxx                    4.0.1                hcfea43d_1  
    libcxxabi                 4.0.1                hcfea43d_1  
    libedit                   3.1.20181209         hb402a30_0  
    libffi                    3.2.1                h475c297_4  
    libgfortran               4.0.0                         2    conda-forge
    liblapack                 3.8.0               14_openblas    conda-forge
    libopenblas               0.3.7                h4bb4525_2    conda-forge
    libprotobuf               3.9.2                hd9629dc_0  
    llvm-openmp               9.0.0                h40edb58_0    conda-forge
    markdown                  3.1.1                    py36_0  
    ncurses                   6.1                  h0a44026_1  
    nltk                      3.4.5                    py36_0  
    numpy                     1.16.4           py36h6b0580a_0    conda-forge
    openssl                   1.1.1d               h0b31af3_0    conda-forge
    pip                       19.3.1                   py36_0  
    pox                       0.2.7                    pypi_0    pypi
    protobuf                  3.9.2            py36h0a44026_0  
    python                    3.6.9                h359304d_0  
    readline                  7.0                  h1de35cc_5  
    setuptools                41.4.0                   py36_0  
    six                       1.12.0                   py36_0  
    sqlite                    3.30.1               ha441bb4_0  
    tensorboard               0.4.0rc3                 py36_2    conda-forge
    tensorflow                1.4.0                    py36_0    conda-forge
    tk                        8.6.8                ha441bb4_0  
    webencodings              0.5.1                    py36_1  
    werkzeug                  0.16.0                     py_0  
    wheel                     0.33.6                   py36_0  
    xz                        5.2.4                h1de35cc_4  
    zlib                      1.2.11               h1de35cc_3 



    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension %  /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/bin/python
    Python 3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 13:42:17) 
    [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import tensorboard
    >>> tensorboard
    <module 'tensorboard' from '/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/__init__.py'>
    >>> 



    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % ls /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/
    __init__.py     backend         db.py           main.py         plugin_util.py  program.py      util.py         webfiles.zip
    __pycache__     data_compat.py  default.py      pip_package     plugins         summary.py      version.py



    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/bin/python /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/main.py 
    /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflo
    w.python.framework.fast_tensor_util' does not match runtime version 3.6
    return f(*args, **kwds)
    Traceback (most recent call last):
    File "/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/main.py", line 67, in <module>
        run_main()
    File "/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/main.py", line 36, in run_main
        tf.app.run(main)
    File "/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorflow/python/platform/app.py", line 48, in run
        _sys.exit(main(_sys.argv[:1] + flags_passthrough))
    File "/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/main.py", line 45, in main
        default.get_assets_zip_provider())
    File "/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/program.py", line 161, in main
        tb = create_tb_app(plugins, assets_zip_provider)
    File "/Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/program.py", line 185, in create_tb_app
        raise ValueError('A logdir must be specified when db is not specified. '
    ValueError: A logdir must be specified when db is not specified. Run `tensorboard --help` for details and examples.




    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % ls tf-log/snrm-extension-example-run-local-2 
    events.out.tfevents.1577297935.Bernhards-MacBook-Pro.local      events.out.tfevents.1577535197.Bernhards-MacBook-Pro.local
    events.out.tfevents.1577473790.Bernhards-MacBook-Pro.local      events.out.tfevents.1577535267.Bernhards-MacBook-Pro.local
    events.out.tfevents.1577473903.Bernhards-MacBook-Pro.local      events.out.tfevents.1577535419.Bernhards-MacBook-Pro.local
    events.out.tfevents.1577474302.Bernhards-MacBook-Pro.local      events.out.tfevents.1577535749.Bernhards-MacBook-Pro.local
    events.out.tfevents.1577474592.Bernhards-MacBook-Pro.local      events.out.tfevents.1577535982.Bernhards-MacBook-Pro.local
    events.out.tfevents.1577475384.Bernhards-MacBook-Pro.local



    (snrm-tf1) bernhardsteindl@Bernhards-MacBook-Pro snrm-extension % /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/bin/python /Users/bernhardsteindl/ana
    conda3/envs/snrm-tf1/lib/python3.6/site-packages/tensorboard/main.py --logdir=/Users/bernhardsteindl/Development/python_workspace/bachelorarbeit/snrm-
    extension/tf-log/snrm-extension-example-run-local-2
    /Users/bernhardsteindl/anaconda3/envs/snrm-tf1/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
    return f(*args, **kwds)
    TensorBoard 0.4.0rc3 at http://Bernhards-MacBook-Pro.local:6006 (Press CTRL+C to quit)



open in browser http://localhost:6006/