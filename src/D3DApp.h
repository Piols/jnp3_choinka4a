#pragma once

#define WIN32_LEAN_AND_MEAN
#define hr(code) {HRESULT hresult = (code); if(FAILED(hresult)) return hresult;}

#include <windows.h>

HRESULT InitDirect3D(HWND hwnd);
//void RecreateRenderTarget(HWND hwnd);
//void DestroyRenderTarget();
HRESULT OnRender(HWND hwnd);
HRESULT OnDestroy(HWND hwnd);