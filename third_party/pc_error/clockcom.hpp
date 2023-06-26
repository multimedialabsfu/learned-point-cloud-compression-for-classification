#ifndef CLOCKCOMM_HPP
#define CLOCKCOMM_HPP

//for time measurement
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#ifndef _WIN32
unsigned long GetTickCount();
#endif


#endif
