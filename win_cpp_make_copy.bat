@echo off
set FILENAME="cauchy_estimator"
set EXE="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\cl.exe"

set INC_MSVC=-I"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include"
set INC_UCRT=-I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\ucrt"
set INC_WINUM=-I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\um"
set INC_WINSHR=-I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.15063.0\shared"

set LIB_PTHREAD=/LIBPATH:"C:\Users\natsn\OneDrive\Desktop\CauchyFriendly\scripts\windows\pthread-win\Pre-built.2\lib\x64" pthreadVC2.lib
set LIB_UUID=/LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.15063.0\um\x64" uuid.lib
set LIB_CPMT=/LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\lib\x64" libcpmt.lib
set LIB_UCRT=/LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.15063.0\ucrt\x64" libucrt.lib

echo "Removing old executable/object files from bin"
del /f ".\bin\%FILENAME%.exe"
del /f ".\bin\%FILENAME%.obj"
echo "Making %FILENAME%.exe located in bin folder..."
%EXE% /O2 /EHsc src/%FILENAME%.cpp %INC_MSVC% %INC_UCRT% %INC_WINUM% %INC_WINSHR% /Fo"bin/%FILENAME%.obj" /link %LIB_PTHREAD% %LIB_UUID% %LIB_CPMT% %LIB_UCRT% /OUT:bin/%FILENAME%.exe
