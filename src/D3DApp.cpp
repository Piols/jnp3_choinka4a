#include <d3d12.h>
#include <directxmath.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include "D3DApp.h"
#include "vertex_shader.h",
#include "pixel_shader.h".

using DirectX::XMFLOAT4X4;
using DirectX::XMFLOAT4;
using DirectX::XMMATRIX;
using namespace DirectX;

#define M_PI 3.14159265358979323846

struct vs_const_buffer_t {
	XMFLOAT4X4 matWorldViewProj;
	XMFLOAT4X4 matWorldView;
	XMFLOAT4X4 matView;

	XMFLOAT4 colMaterial;
	XMFLOAT4 colLight;
	XMFLOAT4 dirLight;
	XMFLOAT4 padding;
};
static_assert(sizeof(vs_const_buffer_t) == 256);

namespace {
    const INT FrameCount = 2;
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<IDXGIFactory7> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Resource> renderTargets[FrameCount];
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12CommandQueue> commandQueue;
    
    ComPtr<ID3D12RootSignature> rootSignature;

    typedef ComPtr<ID3D12DescriptorHeap> HeapType;
    HeapType rtvHeap;
    HeapType cbvHeap;


    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    ComPtr<ID3D12PipelineState> pipelineState;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    UINT rtvDescriptorSize;

    ComPtr<ID3D12Fence> fence;
    UINT frameIndex;
    UINT64 fenceValue;
    HANDLE fenceEvent;

	D3D12_VIEWPORT viewport;

	ComPtr<ID3D12Resource> vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

    D3D12_DESCRIPTOR_RANGE rootDescRange;
    D3D12_ROOT_PARAMETER rootParameter[1];

    ComPtr<ID3D12Resource> vsConstBuffer;
    vs_const_buffer_t vsConstBufferData;
    UINT8* vsConstBufferPointer;

	RECT rc;

    ComPtr<ID3D12Resource> depthBuffer;
	HeapType depthBufferHeap;

	struct vertex_t {
		FLOAT position[3];
		FLOAT normal_vector[3];
		FLOAT color[4];
	};
	size_t const VERTEX_SIZE = sizeof(vertex_t) / sizeof(FLOAT);

	vertex_t flip(vertex_t v) {
		vertex_t res = v;
		res.normal_vector[0] *= -1.f;
		res.normal_vector[1] *= -1.f;
		res.normal_vector[2] *= -1.f;
		return res;
	}

	vertex_t rotate(vertex_t v, float angle) {
		vertex_t res = v;
		res.position[0] = v.position[0] * cos(angle) - v.position[2] * sin(angle);
		res.position[2] = v.position[0] * sin(angle) + v.position[2] * cos(angle);
		res.normal_vector[0] = v.normal_vector[0] * cos(angle) - v.normal_vector[2] * sin(angle);
		res.normal_vector[2] = v.normal_vector[0] * sin(angle) + v.normal_vector[2] * cos(angle);
		return res;
	}

	const int tri_per_level = 11;
	const int levels = 4;
	const float heights[levels + 1] = { -1.0f, 0.0f, 0.6f, 1.f, 1.25f};
	const float wth_ratio = 0.6;
	const FLOAT colors[levels + 1][4] = {
		{1.f, 1.f, 1.f, 1.f},
		{0.f, 0.5f, 0.f, 1.f},
		{0.f, 0.75f, 0.f, 1.f},
		{0.f, 0.9f, 0.f, 1.f},
		{0.f, 1.f, 0.f, 1.f}
	};
	vertex_t triangle_data[tri_per_level * levels * 2 * 3];

	void makeTriangles() {
		for (int level = 0; level < levels; level++) {
			triangle_data[level * tri_per_level * 6] = {
				(heights[level + 1] - heights[level]) * wth_ratio, heights[level], 0.f,
				0.f, 0.f, -1.f,
				colors[level + 1][0], colors[level + 1][1],
				colors[level + 1][2], colors[level + 1][3]
			};
			triangle_data[level * tri_per_level * 6 + 1] = {
				0.f, heights[level], 0.f,
				0.f, 0.f, -1.f,
				colors[level + 1][0], colors[level + 1][1],
				colors[level + 1][2], colors[level + 1][3]
			};

			triangle_data[level * tri_per_level * 6 + 2] = {
				0.f, heights[level + 1], 0.f,
				0.f, 0.f, -1.f,
				colors[0][0], colors[0][1],
				colors[0][2], colors[0][3]
			};
			for (int i = 1; i < tri_per_level; i++) {
				triangle_data[level * tri_per_level * 6 + i * 3] = rotate(triangle_data[level * tri_per_level * 6], i * M_PI * 2 / tri_per_level);
				triangle_data[level * tri_per_level * 6 + i * 3 + 1] = rotate(triangle_data[level * tri_per_level * 6 + 1], i * M_PI * 2 / tri_per_level);
				triangle_data[level * tri_per_level * 6 + i * 3 + 2] = rotate(triangle_data[level * tri_per_level * 6 + 2], i * M_PI * 2 / tri_per_level);
			}
			for (int i = 0; i < tri_per_level; i++) {
				triangle_data[level * tri_per_level * 6 + tri_per_level * 3 + i * 3] = flip(triangle_data[level * tri_per_level * 6 + i * 3 + 1]);
				triangle_data[level * tri_per_level * 6 + tri_per_level * 3 + i * 3 + 1] = flip(triangle_data[level * tri_per_level * 6 + i * 3]);
				triangle_data[level * tri_per_level * 6 + tri_per_level * 3 + i * 3 + 2] = flip(triangle_data[level * tri_per_level * 6 + i * 3 + 2]);
			}
		}
	}

	size_t const NUM_VERTICES = tri_per_level * 6 * levels;;
	size_t const VERTEX_BUFFER_SIZE = NUM_VERTICES * sizeof(vertex_t);
}

void calcNewMatrix() {
	XMMATRIX wvp_matrix;
	static FLOAT angle = 0.0;
	angle += 0.005;

	XMStoreFloat4x4(
		&vsConstBufferData.matWorldView,
		XMMatrixIdentity()
	);

	wvp_matrix = XMMatrixMultiply(
		XMMatrixRotationY(2.5f * angle),	//  @TODO zmienna angle zmienia się
										// o 1 / 64 co ok. 15 ms 
		XMMatrixRotationX(static_cast<FLOAT>(0))//sin(angle)))
	);

	XMStoreFloat4x4(
		&vsConstBufferData.matView,
		wvp_matrix
	);

	wvp_matrix = XMMatrixMultiply(
		wvp_matrix,
		XMMatrixTranslation(0.0f, 0.0f, 4.0f)
	);

	wvp_matrix = XMMatrixMultiply(
		wvp_matrix, 
		XMMatrixPerspectiveFovLH(
			45.0f, viewport.Width / viewport.Height, 1.0f, 100.0f
		)
	);
	wvp_matrix = XMMatrixTranspose(wvp_matrix);
	XMStoreFloat4x4(
		&vsConstBufferData.matWorldViewProj,
		wvp_matrix
	);

	memcpy(
		vsConstBufferPointer, 		// wskaźnik do zmapowanej pamięci (buf. stałego)
		&vsConstBufferData,
		sizeof(vsConstBufferData)
	);

}

HRESULT initDepthBufferAuxData() {
	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {
		.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
		.NumDescriptors = 1,
		.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
		.NodeMask = 0,
	};
	D3D12_HEAP_PROPERTIES heapProp = {
		.Type = D3D12_HEAP_TYPE_DEFAULT,
		.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
		.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,      
		.CreationNodeMask = 1,
		.VisibleNodeMask = 1,
	};
	D3D12_RESOURCE_DESC resDesc = {
		.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		.Alignment = 0,
		.Width = UINT64(rc.right - rc.left), //??? szerokość celu renderowania
		.Height = UINT64(rc.bottom - rc.top), //??? wysokość celu renderowania
		.DepthOrArraySize = 1,
		.MipLevels = 0,
		.Format = DXGI_FORMAT_D32_FLOAT,
		.SampleDesc = {.Count = 1, .Quality = 0 },
		.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
		.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
	};
	//D3D12_CLEAR_VALUE clearValue = {
	//	.Format = DXGI_FORMAT_D32_FLOAT,
	//	.DepthStencil = { .Depth = 1.0f, .Stencil = 0 }
	//};
	D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc = {
		.Format = DXGI_FORMAT_D32_FLOAT,
		.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
		.Flags = D3D12_DSV_FLAG_NONE,
		.Texture2D = {}
	};

	hr(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&depthBufferHeap)));
	hr(device->CreateCommittedResource(
		&heapProp,
		D3D12_HEAP_FLAG_NONE,
		&resDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&depthBuffer))
	);
	device->CreateDepthStencilView(
		depthBuffer.Get(),
		&depthStencilViewDesc,
		depthBufferHeap->GetCPUDescriptorHandleForHeapStart()
	);
}

HRESULT PopulateCommandList(HWND hwnd) {
  hr(commandAllocator->Reset());
  hr(commandList->Reset(commandAllocator.Get(), pipelineState.Get()));

	commandList->SetGraphicsRootSignature(rootSignature.Get());
    
	ID3D12DescriptorHeap* ppHeaps[] = { cbvHeap.Get() };
	commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	
	commandList->SetGraphicsRootDescriptorTable(0, cbvHeap->GetGPUDescriptorHandleForHeapStart());
	
	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &rc);

	D3D12_RESOURCE_BARRIER barrier = {
		.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
		.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
		.Transition = {
			.pResource = renderTargets[frameIndex].Get(),
			.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
			.StateBefore = D3D12_RESOURCE_STATE_PRESENT,
			.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET
		}
    };
    commandList->ResourceBarrier(1, &barrier);

	auto rtvHandleHeapStart = rtvHeap->GetCPUDescriptorHandleForHeapStart();
	rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    
	rtvHandleHeapStart.ptr += frameIndex * rtvDescriptorSize;
	
	auto cpudesc = depthBufferHeap->GetCPUDescriptorHandleForHeapStart();
	commandList->OMSetRenderTargets(
		1, &rtvHandleHeapStart,
		FALSE, 
		&cpudesc
	);

    const float clearColor[] = { 0.0f, 0.8f, 0.8f, 1.0f };
    commandList->ClearRenderTargetView(rtvHandleHeapStart, clearColor, 0, nullptr);
    commandList->ClearDepthStencilView(
		cpudesc,
		D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr
	);

    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
    commandList->DrawInstanced(NUM_VERTICES, NUM_VERTICES / 3, 0, 0);

	D3D12_RESOURCE_BARRIER barrier2 = {
		.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
		.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
		.Transition = {
			.pResource = renderTargets[frameIndex].Get(),
			.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
			.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET,
			.StateAfter = D3D12_RESOURCE_STATE_PRESENT
		}
    };
	commandList->ResourceBarrier(1, &barrier2);

    hr(commandList->Close());
}

HRESULT WaitForPreviousFrame(HWND hwnd) {
    // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
    // This is code implemented as such for simplicity. More advanced samples 
    // illustrate how to use fences for efficient resource usage.

    // Signal and increment the fence value.
    const UINT64 fenceVal = fenceValue;
    hr(commandQueue->Signal(fence.Get(), fenceVal))
    fenceValue++;

    // Wait until the previous frame is finished.
    if (fence->GetCompletedValue() < fenceVal) {
        hr(fence->SetEventOnCompletion(fenceVal, fenceEvent));
        WaitForSingleObject(fenceEvent, INFINITE);
    }

    frameIndex = swapChain->GetCurrentBackBufferIndex();
}

HRESULT InitDirect3D(HWND hwnd) {
	makeTriangles();

	GetClientRect(hwnd, &rc); // @TODO: errors

	hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));

	hr(D3D12CreateDevice(
		nullptr,
		D3D_FEATURE_LEVEL_12_0,
		IID_PPV_ARGS(&device)
	));

	viewport = {
		.TopLeftX = 0.f,
		.TopLeftY = 0.f,
		.Width = FLOAT(rc.right - rc.left),
		.Height = FLOAT(rc.bottom - rc.top),
		.MinDepth = 0.0f,
		.MaxDepth = 1.0f
	};

	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	hr(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = FrameCount;
	swapChainDesc.Width = 0;
	swapChainDesc.Height = 0;
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.SampleDesc.Count = 1;
	ComPtr<IDXGISwapChain1> tempSwapChain;
	hr(factory->CreateSwapChainForHwnd(
		commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
		hwnd,
		&swapChainDesc,
		nullptr,
		nullptr,
		&tempSwapChain
	));

	hr(factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));


	hr(tempSwapChain.As(&swapChain));

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = FrameCount;
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc = {
        .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        .NumDescriptors = 1,
        .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        .NodeMask = 0  
    };

		hr(device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)));
		hr(device->CreateDescriptorHeap(&cbvHeapDesc, IID_PPV_ARGS(&cbvHeap)));
    
	rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
	for (UINT n = 0; n < FrameCount; n++) {
		hr(swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
		device->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
		rtvHandle.ptr += rtvDescriptorSize;
	}

	hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));

	hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&commandList)));
	hr(commandList->Close());

	// assets:

    D3D12_HEAP_PROPERTIES vsHeapTypeProp = {
        .Type = D3D12_HEAP_TYPE_UPLOAD,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1,
    };
    D3D12_RESOURCE_DESC vsHeapResourceDesc =  {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = 256,		// @TODO rozmiar bufora stałego (w bajtach), musi być 
                        // wielokrotnością 256 bajtów
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = { .Count = 1, .Quality = 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = D3D12_RESOURCE_FLAG_NONE,
    };

		hr(device->CreateCommittedResource(
			&vsHeapTypeProp,
			D3D12_HEAP_FLAG_NONE,
			&vsHeapResourceDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&vsConstBuffer))
		);

    D3D12_CONSTANT_BUFFER_VIEW_DESC vbViewDesc = {
        .BufferLocation = vsConstBuffer->GetGPUVirtualAddress(),
        .SizeInBytes = 256, // @???
    };
    device->CreateConstantBufferView(&vbViewDesc, cbvHeap->GetCPUDescriptorHandleForHeapStart());

    rootDescRange = {
        .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
        .NumDescriptors = 1,
        .BaseShaderRegister = 0,
        .RegisterSpace = 0,
        .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };


    rootParameter[0] = {
        .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        .DescriptorTable = { 1, &rootDescRange},	// adr. rekordu poprzedniego typu
        .ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX,
    };


	D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {
		.NumParameters = _countof(rootParameter),
		.pParameters = rootParameter,
		.NumStaticSamplers = 0,
		.pStaticSamplers = nullptr,
		.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS,
	};

	ComPtr<ID3DBlob> signature;
	ComPtr<ID3DBlob> error;
	hr(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
	hr(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature)));

	D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
		{
			.SemanticName = "POSITION",
			.SemanticIndex = 0,
			.Format = DXGI_FORMAT_R32G32B32_FLOAT,
			.InputSlot = 0,
			.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT,
			.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
			.InstanceDataStepRate = 0
		},
		{
			.SemanticName = "NORMAL",
			.SemanticIndex = 0,
			.Format = DXGI_FORMAT_R32G32B32_FLOAT,
			.InputSlot = 0,
			.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT,
			.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
			.InstanceDataStepRate = 0
		},
		{
			.SemanticName = "COLOR",
			.SemanticIndex = 0,
			.Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
			.InputSlot = 0,
			.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT,
			.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
			.InstanceDataStepRate = 0
		}
	};

	D3D12_BLEND_DESC blendDesc = {
		.AlphaToCoverageEnable = FALSE,
		.IndependentBlendEnable = FALSE,
		.RenderTarget = {
			{
			 .BlendEnable = FALSE,
			 .LogicOpEnable = FALSE,
			 .SrcBlend = D3D12_BLEND_ONE,
			 .DestBlend = D3D12_BLEND_ZERO,
			 .BlendOp = D3D12_BLEND_OP_ADD,
			 .SrcBlendAlpha = D3D12_BLEND_ONE,
			 .DestBlendAlpha = D3D12_BLEND_ZERO,
			 .BlendOpAlpha = D3D12_BLEND_OP_ADD,
			 .LogicOp = D3D12_LOGIC_OP_NOOP,
			 .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL
			}
		}
	};

	D3D12_RASTERIZER_DESC rasterizerDesc = {
		.FillMode = D3D12_FILL_MODE_SOLID,
		.CullMode = D3D12_CULL_MODE_BACK,
		.FrontCounterClockwise = FALSE,
		.DepthBias = D3D12_DEFAULT_DEPTH_BIAS,
		.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
		.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
		.DepthClipEnable = TRUE,
		.MultisampleEnable = FALSE,
		.AntialiasedLineEnable = FALSE,
		.ForcedSampleCount = 0,
		.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
	};

	D3D12_DEPTH_STENCIL_DESC depthStencilDesc = {
		.DepthEnable = TRUE,
		.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL,
		.DepthFunc = D3D12_COMPARISON_FUNC_LESS,
		.StencilEnable = FALSE,
		.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK,
		.StencilWriteMask = D3D12_DEFAULT_STENCIL_READ_MASK,
		.FrontFace = {
			.StencilFailOp = D3D12_STENCIL_OP_KEEP,
			.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP,
			.StencilPassOp = D3D12_STENCIL_OP_KEEP,
			.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS
		},
		.BackFace = {
			.StencilFailOp = D3D12_STENCIL_OP_KEEP,
			.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP,
			.StencilPassOp = D3D12_STENCIL_OP_KEEP,
			.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS
		}
	};

	D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineStateDesc = {
		.DepthStencilState = depthStencilDesc,
		.DSVFormat = DXGI_FORMAT_D32_FLOAT,
	};

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
		.pRootSignature = rootSignature.Get(),
		.VS = { vs_main, sizeof(vs_main) },
		.PS = { ps_main, sizeof(ps_main) },
		.BlendState = blendDesc,
		.SampleMask = UINT_MAX,
		.RasterizerState = rasterizerDesc,
		.DepthStencilState = depthStencilDesc,
		.InputLayout = { inputElementDescs, _countof(inputElementDescs) },
		.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
		.NumRenderTargets = 1,
		.DSVFormat = DXGI_FORMAT_D32_FLOAT,
		.SampleDesc = {.Count = 1},
	};
	psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

	hr(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState)));

	// Create the vertex buffer.
    {
        // Note: using upload heaps to transfer static data like vert buffers is not 
        // recommended. Every time the GPU needs it, the upload heap will be marshalled 
        // over. Please read up on Default Heap usage. An upload heap is used here for 
        // code simplicity and because there are very few verts to actually transfer.
        D3D12_HEAP_PROPERTIES heapProps({
            .Type = D3D12_HEAP_TYPE_UPLOAD,
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,            .CreationNodeMask = 1,
            .VisibleNodeMask = 1,
        });

        D3D12_RESOURCE_DESC desc({
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Alignment = 0,
            .Width = VERTEX_BUFFER_SIZE,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            .Flags = D3D12_RESOURCE_FLAG_NONE,
        });
   
				hr(device->CreateCommittedResource(
					&heapProps,
					D3D12_HEAP_FLAG_NONE,
					&desc,
					D3D12_RESOURCE_STATE_GENERIC_READ,
					nullptr,
					IID_PPV_ARGS(&vertexBuffer))
				);

        // Copy the triangle data to the vertex buffer.
        UINT8* pVertexDataBegin;
        D3D12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
				hr(vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, triangle_data, sizeof(triangle_data));
        vertexBuffer->Unmap(0, nullptr);

        // Initialize the vertex buffer view.
        vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        vertexBufferView.StrideInBytes = sizeof(vertex_t);
        vertexBufferView.SizeInBytes = VERTEX_BUFFER_SIZE;
    }

	hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
	fenceValue = 1;

	// Create an event handle to use for frame synchronization.
	fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (fenceEvent == nullptr) {
		hr(HRESULT_FROM_WIN32(GetLastError()));
	}

	DirectX::XMStoreFloat4x4(&vsConstBufferData.matWorldViewProj, DirectX::XMMatrixIdentity());
    D3D12_RANGE constBufferDataRange = {0, 0};
	hr(vsConstBuffer->Map(0, &constBufferDataRange, reinterpret_cast<void**>(&vsConstBufferPointer)));
   
	memcpy(vsConstBufferPointer, &vsConstBufferData, sizeof(vsConstBufferData));

	hr(initDepthBufferAuxData());

	vsConstBufferData.colLight = {1, 1, 1, 1};
	vsConstBufferData.dirLight = {0.0, 0.0, 1.0, 1};
	vsConstBufferData.colMaterial = {0.4, 1, 0.4, 1};

	// Wait for the command list to execute; we are reusing the same command 
	// list in our main loop but for now, we just want to wait for setup to 
	// complete before continuing.
	hr(WaitForPreviousFrame(hwnd));
}

HRESULT OnRender(HWND hwnd) {

	// @TODO: to ma być w jakimś update
	calcNewMatrix();

  // Record all the commands we need to render the scene into the command list.
  hr(PopulateCommandList(hwnd));

  // Execute the command list.
  ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
  commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // Present the frame.
  hr(swapChain->Present(1, 0));

  hr(WaitForPreviousFrame(hwnd));
}

HRESULT OnDestroy(HWND hwnd) {

    // Wait for the GPU to be done with all resources.
    hr(WaitForPreviousFrame(hwnd));

    CloseHandle(fenceEvent);
}