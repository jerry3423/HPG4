#include "vulkan_window.hpp"

#include <tuple>
#include <limits>
#include <vector>
#include <utility>
#include <optional>
#include <algorithm>
#include <unordered_set>

#include <cstdio>
#include <cassert>
#include <vulkan/vulkan_core.h>

#include "error.hpp"
#include "to_string.hpp"
#include "context_helpers.hxx"
namespace lut = labutils;


namespace
{
	// The device selection process has changed somewhat w.r.t. the one used 
	// earlier (e.g., with VulkanContext.
	VkPhysicalDevice select_device( VkInstance, VkSurfaceKHR );
	float score_device( VkPhysicalDevice, VkSurfaceKHR );

	std::optional<std::uint32_t> find_queue_family( VkPhysicalDevice, VkQueueFlags, VkSurfaceKHR = VK_NULL_HANDLE );

	VkDevice create_device( 
		VkPhysicalDevice,
		std::vector<std::uint32_t> const& aQueueFamilies,
		std::vector<char const*> const& aEnabledDeviceExtensions = {}
	);

	std::vector<VkSurfaceFormatKHR> get_surface_formats( VkPhysicalDevice, VkSurfaceKHR );
	std::unordered_set<VkPresentModeKHR> get_present_modes( VkPhysicalDevice, VkSurfaceKHR );

	std::tuple<VkSwapchainKHR,VkFormat,VkExtent2D> create_swapchain(
		VkPhysicalDevice,
		VkSurfaceKHR,
		VkDevice,
		GLFWwindow*,
		std::vector<std::uint32_t> const& aQueueFamilyIndices = {},
		VkSwapchainKHR aOldSwapchain = VK_NULL_HANDLE
	);

	void get_swapchain_images( VkDevice, VkSwapchainKHR, std::vector<VkImage>& );
	void create_swapchain_image_views( VkDevice, VkFormat, std::vector<VkImage> const&, std::vector<VkImageView>& );
}

namespace labutils
{
	// VulkanWindow
	VulkanWindow::VulkanWindow() = default;

	VulkanWindow::~VulkanWindow()
	{
		// Device-related objects
		for (auto const view : swapViews)
			vkDestroyImageView(device, view, nullptr);

		if (VK_NULL_HANDLE != swapchain)
			vkDestroySwapchainKHR(device, swapchain, nullptr);

		// Window and related objects
		if (VK_NULL_HANDLE != surface)
			vkDestroySurfaceKHR(instance, surface, nullptr);

		if (window)
		{
			glfwDestroyWindow(window);

			// The following assumes that we never create more than one window;
			// if there are multiple windows, destroying one of them would
			// unload the whole GLFW library. Nevertheless, this solution is
			// convenient when only dealing with one window (which we will do
			// in the exercises), as it ensure that GLFW is unloaded after all
			// window-related resources are.
			glfwTerminate();
		}
	}

	VulkanWindow::VulkanWindow(VulkanWindow&& aOther) noexcept
		: VulkanContext(std::move(aOther))
		, window(std::exchange(aOther.window, VK_NULL_HANDLE))
		, surface(std::exchange(aOther.surface, VK_NULL_HANDLE))
		, presentFamilyIndex(aOther.presentFamilyIndex)
		, presentQueue(std::exchange(aOther.presentQueue, VK_NULL_HANDLE))
		, swapchain(std::exchange(aOther.swapchain, VK_NULL_HANDLE))
		, swapImages(std::move(aOther.swapImages))
		, swapViews(std::move(aOther.swapViews))
		, swapchainFormat(aOther.swapchainFormat)
		, swapchainExtent(aOther.swapchainExtent)
	{}

	VulkanWindow& VulkanWindow::operator=(VulkanWindow&& aOther) noexcept
	{
		VulkanContext::operator=(std::move(aOther));
		std::swap(window, aOther.window);
		std::swap(surface, aOther.surface);
		std::swap(presentFamilyIndex, aOther.presentFamilyIndex);
		std::swap(presentQueue, aOther.presentQueue);
		std::swap(swapchain, aOther.swapchain);
		std::swap(swapImages, aOther.swapImages);
		std::swap(swapViews, aOther.swapViews);
		std::swap(swapchainFormat, aOther.swapchainFormat);
		std::swap(swapchainExtent, aOther.swapchainExtent);
		return *this;
	}

	// make_vulkan_window()
	VulkanWindow make_vulkan_window()
	{
		VulkanWindow ret;

		// Initialize Volk
		if (auto const res = volkInitialize(); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to load Vulkan API\n"
				"Volk returned error %s", lut::to_string(res).c_str()
			);
		}

		//initialize GLFW
		if (glfwInit() != GLFW_TRUE)
		{
			char const* msg = nullptr;
			glfwGetError(&msg);
			throw lut::Error("GLFW initialized failed: %s", msg);
		}
		if (!glfwVulkanSupported())
		{
			throw lut::Error("GLFW is not support Vulkan");
		}

		// Check for instance layers and extensions
		auto const supportedLayers = detail::get_instance_layers();
		auto const supportedExtensions = detail::get_instance_extensions();

		bool enableDebugUtils = false;

		std::vector<char const*> enabledLayers, enabledExensions;

		//check that the instance extensions required by GLFW are available
		std::uint32_t glfwReqExtensionCount;
		char const** glfwReqExtensions = glfwGetRequiredInstanceExtensions(&glfwReqExtensionCount);
		for (std::uint32_t i = 0; i < glfwReqExtensionCount; ++i)
		{
			if (!supportedExtensions.count(glfwReqExtensions[i]))
			{
				throw lut::Error("GLFW required instance extension %s is not available", glfwReqExtensions[i]);
			}
			enabledExensions.emplace_back(glfwReqExtensions[i]);
		}

		// Validation layers support.
#		if !defined(NDEBUG) // debug builds only
		if (supportedLayers.count("VK_LAYER_KHRONOS_validation"))
		{
			enabledLayers.emplace_back("VK_LAYER_KHRONOS_validation");
		}

		if (supportedExtensions.count("VK_EXT_debug_utils"))
		{
			enableDebugUtils = true;
			enabledExensions.emplace_back("VK_EXT_debug_utils");
		}
#		endif // ~ debug builds

		for (auto const& layer : enabledLayers)
			std::fprintf(stderr, "Enabling layer: %s\n", layer);

		for (auto const& extension : enabledExensions)
			std::fprintf(stderr, "Enabling instance extension: %s\n", extension);

		// Create Vulkan instance
		ret.instance = detail::create_instance(enabledLayers, enabledExensions, enableDebugUtils);

		// Load rest of the Vulkan API
		volkLoadInstance(ret.instance);

		// Setup debug messenger
		if (enableDebugUtils)
			ret.debugMessenger = detail::create_debug_messenger(ret.instance);

		//create GLFW window
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);//only for rendering
		ret.window = glfwCreateWindow(1280, 720, "Coursework4", nullptr, nullptr);
		if (!ret.window)
		{
			char const* msg = nullptr;
			glfwGetError(&msg);

			throw lut::Error("Unable to create GLFW window\n""error = %s", msg);
		}

		//get VkSurfaceKHR from the window
		if (auto const res = glfwCreateWindowSurface(ret.instance, ret.window, nullptr, &ret.surface); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create VkSurfaceKHR\n""glfwCreateWindowSurface() returned %s", lut::to_string(res).c_str());
		}

		// Select appropriate Vulkan device
		ret.physicalDevice = select_device(ret.instance, ret.surface);
		if (VK_NULL_HANDLE == ret.physicalDevice)
			throw lut::Error("No suitable physical device found!");

		{
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(ret.physicalDevice, &props);
			std::fprintf(stderr, "Selected device: %s (%d.%d.%d)\n", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
		}

		// Create a logical device
		// Enable required extensions. The device selection method ensures that
		// the VK_KHR_swapchain extension is present, so we can safely just
		// request it without further checks.
		std::vector<char const*> enabledDevExensions;

		enabledDevExensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		for (auto const& ext : enabledDevExensions)
			std::fprintf(stderr, "Enabling device extension: %s\n", ext);

		// We need one or two queues:
		// - best case: one GRAPHICS queue that can present
		// - otherwise: one GRAPHICS queue and any queue that can present
		std::vector<std::uint32_t> queueFamilyIndices;

		//logic to select necessary queue families to instantiate
		if (auto const index = find_queue_family(ret.physicalDevice, VK_QUEUE_GRAPHICS_BIT, ret.surface))
		{
			ret.graphicsFamilyIndex = *index;
			queueFamilyIndices.emplace_back(*index);
		}
		else
		{
			auto graphics = find_queue_family(ret.physicalDevice, VK_QUEUE_GRAPHICS_BIT);
			auto present = find_queue_family(ret.physicalDevice, 0, ret.surface);

			assert(graphics && present);

			ret.graphicsFamilyIndex = *graphics;
			ret.presentFamilyIndex = *present;

			queueFamilyIndices.emplace_back(*graphics);
			queueFamilyIndices.emplace_back(*present);
		}

		ret.device = create_device(ret.physicalDevice, queueFamilyIndices, enabledDevExensions);

		// Retrieve VkQueues
		vkGetDeviceQueue(ret.device, ret.graphicsFamilyIndex, 0, &ret.graphicsQueue);

		assert(VK_NULL_HANDLE != ret.graphicsQueue);

		if (queueFamilyIndices.size() >= 2)
			vkGetDeviceQueue(ret.device, ret.presentFamilyIndex, 0, &ret.presentQueue);
		else
		{
			ret.presentFamilyIndex = ret.graphicsFamilyIndex;
			ret.presentQueue = ret.graphicsQueue;
		}

		// Create swap chain
		std::tie(ret.swapchain, ret.swapchainFormat, ret.swapchainExtent) = create_swapchain(ret.physicalDevice, ret.surface, ret.device, ret.window, queueFamilyIndices);

		// Get swap chain images & create associated image views
		get_swapchain_images(ret.device, ret.swapchain, ret.swapImages);
		create_swapchain_image_views(ret.device, ret.swapchainFormat, ret.swapImages, ret.swapViews);

		// Done
		return ret;
	}

	SwapChanges recreate_swapchain(VulkanWindow& aWindow)
	{
		auto const oldFormat = aWindow.swapchainFormat;
		auto const oldExtent = aWindow.swapchainExtent;

		VkSwapchainKHR oldSwapchain = aWindow.swapchain;

		//destroy old image view
		for (auto view : aWindow.swapViews)
		{
			vkDestroyImageView(aWindow.device, view, nullptr);
		}

		aWindow.swapViews.clear();
		aWindow.swapImages.clear();

		std::vector<std::uint32_t> queueFamilyIndices;
		if (aWindow.presentFamilyIndex != aWindow.graphicsFamilyIndex)
		{
			queueFamilyIndices.emplace_back(aWindow.graphicsFamilyIndex);
			queueFamilyIndices.emplace_back(aWindow.presentFamilyIndex);
		}

		try
		{
			//recreate swapchain
			std::tie(aWindow.swapchain, aWindow.swapchainFormat, aWindow.swapchainExtent) =
				create_swapchain(aWindow.physicalDevice, aWindow.surface, aWindow.device, aWindow.window, queueFamilyIndices, oldSwapchain);
		}
		catch (...)
		{
			aWindow.swapchain = oldSwapchain;
			throw;
		}

		//destroy old swapchain
		vkDestroySwapchainKHR(aWindow.device, oldSwapchain, nullptr);

		//get new image view
		get_swapchain_images(aWindow.device, aWindow.swapchain, aWindow.swapImages);
		create_swapchain_image_views(aWindow.device, aWindow.swapchainFormat, aWindow.swapImages, aWindow.swapViews);

		SwapChanges ret{};

		if (oldExtent.width != aWindow.swapchainExtent.width || oldExtent.height != aWindow.swapchainExtent.height)
			ret.changedSize = true;
		if (oldFormat != aWindow.swapchainFormat)
			ret.changedFormat = true;
		return ret;
	}
}

namespace
{
	std::vector<VkSurfaceFormatKHR> get_surface_formats(VkPhysicalDevice aPhysicalDev, VkSurfaceKHR aSurface)
	{
		uint32_t formatCount;
		std::vector<VkSurfaceFormatKHR> surfaceFormats;
		vkGetPhysicalDeviceSurfaceFormatsKHR(aPhysicalDev, aSurface, &formatCount, nullptr);
		surfaceFormats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(aPhysicalDev, aSurface, &formatCount, surfaceFormats.data());
		return surfaceFormats;
	}

	std::unordered_set<VkPresentModeKHR> get_present_modes(VkPhysicalDevice aPhysicalDev, VkSurfaceKHR aSurface)
	{
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(aPhysicalDev, aSurface, &presentModeCount, nullptr);

		std::vector<VkPresentModeKHR> presentModes(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(aPhysicalDev, aSurface, &presentModeCount, presentModes.data());
		std::unordered_set<VkPresentModeKHR> presentModeSet;
		for (const auto& presentMode : presentModes) {
			presentModeSet.insert(presentMode);
		}
		return presentModeSet;
	}

	std::tuple<VkSwapchainKHR, VkFormat, VkExtent2D> create_swapchain(VkPhysicalDevice aPhysicalDev, VkSurfaceKHR aSurface, VkDevice aDevice, GLFWwindow* aWindow, std::vector<std::uint32_t> const& aQueueFamilyIndices, VkSwapchainKHR aOldSwapchain)
	{
		auto const formats = get_surface_formats(aPhysicalDev, aSurface);
		auto const modes = get_present_modes(aPhysicalDev, aSurface);

		//pick appropriate VkSurfaceFormatKHR format.
		assert(!formats.empty());
		VkSurfaceFormatKHR format = formats[0];
		for (auto const fmt : formats)
		{
			if (VK_FORMAT_R8G8B8A8_SRGB == fmt.format && VK_COLOR_SPACE_SRGB_NONLINEAR_KHR == fmt.colorSpace)
			{
				format = fmt;
				break;
			}
			if (VK_FORMAT_B8G8R8A8_SRGB == fmt.format && VK_COLOR_SPACE_SRGB_NONLINEAR_KHR == fmt.colorSpace)
			{
				format = fmt;
				break;
			}
		}
		//pick appropriate VkPresentModeKHR
		VkPresentModeKHR presentMode = VK_PRESENT_MODE_MAILBOX_KHR;

		//pick image count
		VkSurfaceCapabilitiesKHR caps;
		if (auto const res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(aPhysicalDev, aSurface, &caps); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to get surface capabilities\n""vkGetPhysicalDeviceSurfaceCapabilitiesKHR() returned %s", lut::to_string(res).c_str());
		}
		std::uint32_t imageCount = 2;
		if (imageCount < caps.minImageCount + 1)
			imageCount = caps.minImageCount + 1;

		if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
			imageCount = caps.maxImageCount;

		//figure out swap extent
		VkExtent2D extent = caps.currentExtent;
		if (std::numeric_limits<std::uint32_t>::max() == extent.width)
		{
			int width, height;
			glfwGetFramebufferSize(aWindow, &width, &height);
			auto const& min = caps.minImageExtent;
			auto const& max = caps.maxImageExtent;

			extent.width = std::clamp(std::uint32_t(width), min.width, max.width);
			extent.height = std::clamp(std::uint32_t(height), min.height, max.height);
		}

		// create swap chain
		VkSwapchainCreateInfoKHR chainInfo{};
		chainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		chainInfo.surface = aSurface;
		chainInfo.minImageCount = imageCount;
		chainInfo.imageFormat = format.format;
		chainInfo.imageColorSpace = format.colorSpace;
		chainInfo.imageExtent = extent;
		chainInfo.imageArrayLayers = 1;
		chainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		chainInfo.preTransform = caps.currentTransform;
		chainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		chainInfo.presentMode = presentMode;
		chainInfo.clipped = VK_TRUE;
		chainInfo.oldSwapchain = aOldSwapchain;
		if (aQueueFamilyIndices.size() <= 1)
		{
			chainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}
		else
		{
			chainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			chainInfo.queueFamilyIndexCount = std::uint32_t(aQueueFamilyIndices.size());
			chainInfo.pQueueFamilyIndices = aQueueFamilyIndices.data();
		}
		VkSwapchainKHR chain = VK_NULL_HANDLE;
		if (auto const res = vkCreateSwapchainKHR(aDevice, &chainInfo, nullptr, &chain); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create swap chain\n""vkCreateSwapchainKHR() returned %s", lut::to_string(res).c_str());
		}

		return{ chain, format.format, extent };
	}


	void get_swapchain_images(VkDevice aDevice, VkSwapchainKHR aSwapchain, std::vector<VkImage>& aImages)
	{
		//get swapchain image handles with vkGetSwapchainImagesKHR
		uint32_t imageSize;
		vkGetSwapchainImagesKHR(aDevice, aSwapchain, &imageSize, NULL);
		aImages.resize(imageSize);
		vkGetSwapchainImagesKHR(aDevice, aSwapchain, &imageSize, aImages.data());
	}

	void create_swapchain_image_views(VkDevice aDevice, VkFormat aSwapchainFormat, std::vector<VkImage> const& aImages, std::vector<VkImageView>& aViews)
	{
		//create a VkImageView for each of the VkImages.
		for (std::size_t i = 0; i < aImages.size(); ++i)
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = aImages[i];
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = aSwapchainFormat;
			viewInfo.components = VkComponentMapping{
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY
			};
			viewInfo.subresourceRange = VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, 1,
				0, 1
			};
			VkImageView view = VK_NULL_HANDLE;
			if (auto const res = vkCreateImageView(aDevice, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create image view for swap chain image %zu\n""vkCreateImageView() returned %s", i, lut::to_string(res).c_str());
			}
			aViews.emplace_back(view);
		}
	}
}

namespace
{
	// Note: this finds *any* queue that supports the aQueueFlags. As such,
	//   find_queue_family( ..., VK_QUEUE_TRANSFER_BIT, ... );
	// might return a GRAPHICS queue family, since GRAPHICS queues typically
	// also set TRANSFER (and indeed most other operations; GRAPHICS queues are
	// required to support those operations regardless). If you wanted to find
	// a dedicated TRANSFER queue (e.g., such as those that exist on NVIDIA
	// GPUs), you would need to use different logic.
	std::optional<std::uint32_t> find_queue_family(VkPhysicalDevice aPhysicalDev, VkQueueFlags aQueueFlags, VkSurfaceKHR aSurface)
	{
		//find queue family with the specified queue flags that can  present to the surface
		std::uint32_t numQueues = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(aPhysicalDev, &numQueues, nullptr);
		std::vector<VkQueueFamilyProperties> families(numQueues);
		vkGetPhysicalDeviceQueueFamilyProperties(aPhysicalDev, &numQueues, families.data());
		for (std::uint32_t i = 0; i < numQueues; ++i)
		{
			auto const& family = families[i];
			if (aQueueFlags == (aQueueFlags & family.queueFlags))
			{
				if (VK_NULL_HANDLE == aSurface)
				{
					return i;
				}
				VkBool32 supported = VK_FALSE;
				auto const res = vkGetPhysicalDeviceSurfaceSupportKHR(aPhysicalDev, i, aSurface, &supported);
				if (VK_SUCCESS == res && supported)
					return i;
			}
		}
		return std::nullopt;
	}

	VkDevice create_device(VkPhysicalDevice aPhysicalDev, std::vector<std::uint32_t> const& aQueues, std::vector<char const*> const& aEnabledExtensions)
	{
		if (aQueues.empty())
			throw lut::Error("create_device(): no queues requested");

		float queuePriorities[1] = { 1.f };

		std::vector<VkDeviceQueueCreateInfo> queueInfos(aQueues.size());
		for (std::size_t i = 0; i < aQueues.size(); ++i)
		{
			auto& queueInfo = queueInfos[i];
			queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueInfo.queueFamilyIndex = aQueues[i];
			queueInfo.queueCount = 1;
			queueInfo.pQueuePriorities = queuePriorities;
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		//cheeck if device support sampler anisotropy
		vkGetPhysicalDeviceFeatures(aPhysicalDev, &deviceFeatures);
		if (!deviceFeatures.samplerAnisotropy) {
			throw lut::Error("Device does not support sampler anisotropy!");
		}
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo deviceInfo{};
		deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		deviceInfo.queueCreateInfoCount = std::uint32_t(queueInfos.size());
		deviceInfo.pQueueCreateInfos = queueInfos.data();

		deviceInfo.enabledExtensionCount = std::uint32_t(aEnabledExtensions.size());
		deviceInfo.ppEnabledExtensionNames = aEnabledExtensions.data();

		deviceInfo.pEnabledFeatures = &deviceFeatures;

		VkDevice device = VK_NULL_HANDLE;
		if (auto const res = vkCreateDevice(aPhysicalDev, &deviceInfo, nullptr, &device); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create logical device\n"
				"vkCreateDevice() returned %s", lut::to_string(res).c_str()
			);
		}

		return device;
	}
}

namespace
{
	float score_device(VkPhysicalDevice aPhysicalDev, VkSurfaceKHR aSurface)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(aPhysicalDev, &props);

		// Only consider Vulkan 1.1 devices
		auto const major = VK_API_VERSION_MAJOR(props.apiVersion);
		auto const minor = VK_API_VERSION_MINOR(props.apiVersion);

		if (major < 1 || (major == 1 && minor < 2))
		{
			std::fprintf(stderr, "Info: Discarding device '%s': insufficient vulkan version\n", props.deviceName);
			return -1.f;
		}

		//additional checks
		//check that the VK_KHR_swapchain extension is supported
		auto const exts = lut::detail::get_device_extensions(aPhysicalDev);
		if (!exts.count(VK_KHR_SWAPCHAIN_EXTENSION_NAME))
		{
			std::fprintf(stderr, "Info: Discarding device ¡¯%s¡¯: extension %s missing\n", props.deviceName, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
			return -1.f;
		}
		//check that there is a queue family that can present to the given surface
		if (!find_queue_family(aPhysicalDev, 0, aSurface))
		{
			std::fprintf(stderr, "Info: Discarding device ¡¯%s¡¯: can¡¯t present to surface\n", props.deviceName);
			return -1.f;
		}
		//check that there is a queue family that supports graphics commands
		if (!find_queue_family(aPhysicalDev, VK_QUEUE_GRAPHICS_BIT))
		{
			std::fprintf(stderr, "Info: Discarding device ¡¯%s¡¯: no graphics queue family\n", props.deviceName);
			return -1.f;
		}

		// Discrete GPU > Integrated GPU > others
		float score = 0.f;

		if (VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU == props.deviceType)
			score += 500.f;
		else if (VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == props.deviceType)
			score += 100.f;

		return score;
	}

	VkPhysicalDevice select_device(VkInstance aInstance, VkSurfaceKHR aSurface)
	{
		std::uint32_t numDevices = 0;
		if (auto const res = vkEnumeratePhysicalDevices(aInstance, &numDevices, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to get physical device count\n"
				"vkEnumeratePhysicalDevices() returned %s", lut::to_string(res).c_str()
			);
		}

		std::vector<VkPhysicalDevice> devices(numDevices, VK_NULL_HANDLE);
		if (auto const res = vkEnumeratePhysicalDevices(aInstance, &numDevices, devices.data()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to get physical device list\n"
				"vkEnumeratePhysicalDevices() returned %s", lut::to_string(res).c_str()
			);
		}

		float bestScore = -1.f;
		VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

		for (auto const device : devices)
		{
			auto const score = score_device(device, aSurface);
			if (score > bestScore)
			{
				bestScore = score;
				bestDevice = device;
			}
		}

		return bestDevice;
	}
}

