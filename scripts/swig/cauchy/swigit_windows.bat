@echo off
:: Batch script to run SWIG, compile with cl.exe, and link with link.exe
echo "Modify the contents of this file judiciously"

set FILE_NAME=pycauchy
set SWIG_FILE=%FILE_NAME%.i
:: Remove old files
del /f _%FILE_NAME%.pyd
del /f _%FILE_NAME%.exp
del /f _%FILE_NAME%.lib
del /f %FILE_NAME%_wrap.cxx
del /f %FILE_NAME%_wrap.obj
del /f %FILE_NAME%.py
::rmdir /s /q __pycache__
echo "All temp files / libraries initially deleted"

:: Define paths and flags
set MY_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\cl.exe"
set MY_LINK="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\link.exe"
set MY_SWIG_EXE="C:\Users\natsn\OneDrive\Desktop\CauchyFriendly\scripts\windows\swigwin-4.2.1\swig.exe"

:: Include Headers
set INC_PYTHON=-I"C:\Users\natsn\AppData\Local\Programs\Python\Python37\include"
set INC_NUMPY=-I"C:\Users\natsn\AppData\Local\Programs\Python\Python37\lib\site-packages\numpy\core\include"
set INC_MSVC=-I"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include"
set INC_UCRT=-I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\ucrt"
set INC_WINUM=-I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\um"
set INC_WINSHR=-I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\shared"

:: Libraries
set LIB_PYTHON=/LIBPATH:"C:\Users\natsn\AppData\Local\Programs\Python\Python37\libs" libpython37.a
set LIB_UCRT=/LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.15063.0\ucrt\x64" libucrt.lib
set LIB_UM=/LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.15063.0\um\x64" uuid.lib
set LIB_PTHREAD=/LIBPATH:"C:\Users\natsn\OneDrive\Desktop\CauchyFriendly\scripts\windows\pthread-win\Pre-built.2\lib\x64" pthreadVC2.lib
set LIB_MSVC=/LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\lib\x64" msvcprt.lib

:: Run SWIG to generate wrapper code
%MY_SWIG_EXE% -c++ -python %FILE_NAME%.i
if errorlevel 1 (
    echo "[ERROR:] SWIG command failed!"
    exit /b 1
)

:: Compile the generated wrapper code with cl.exe
%MY_EXE% /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS -c %FILE_NAME%_wrap.cxx %INC_PYTHON% %INC_UCRT% %INC_MSVC% %INC_WINSHR% %INC_WINUM% %INC_NUMPY%
if errorlevel 1 (
    echo "[ERROR:] Compilation with cl.exe failed!"
    exit /b 1
)

:: Link the object file with link.exe to create the Python extension
%MY_LINK% %LIB_PYTHON% %LIB_PTHREAD% %LIB_MSVC% %LIB_UM% %LIB_UCRT% %FILE_NAME%_wrap.obj /DLL /OUT:_%FILE_NAME%.pyd
if errorlevel 1 (
    echo "[ERROR:] Linking with link.exe failed!"
    exit /b 1
)

echo "[SUCCESS:] Python module _%FILE_NAME%.pyd created successfully!"
echo "All temp files / libraries (re)created!"
echo "Module %FILE_NAME%.py is now ready for use!"