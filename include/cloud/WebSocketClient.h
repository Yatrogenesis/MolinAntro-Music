/**
 * @file WebSocketClient.h
 * @brief REAL WebSocket client implementation using websocketpp
 *
 * This is PRODUCTION-READY networking code with:
 * - Actual TCP/TLS connections
 * - Binary frame support for audio streaming
 * - Automatic reconnection with exponential backoff
 * - Thread-safe message queues
 *
 * Dependencies: websocketpp (header-only), asio (standalone)
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#pragma once

// Use standalone Asio (no Boost dependency)
#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_THREAD_

#ifdef _WIN32
#define _WIN32_WINNT 0x0601  // Windows 7+
#endif

#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

#include <string>
#include <functional>
#include <memory>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>

namespace MolinAntro {
namespace Cloud {
namespace Net {

/**
 * @brief Connection state machine states
 */
enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed
};

/**
 * @brief WebSocket message type codes for MolinAntro protocol
 */
enum class MessageType : uint8_t {
    // Control messages (JSON payload)
    Authenticate = 0x01,
    AuthResult = 0x02,
    PresenceUpdate = 0x10,
    PresenceBroadcast = 0x11,
    ProjectChange = 0x20,
    ChangeAck = 0x21,
    ChangeReject = 0x22,
    ConflictNotify = 0x23,
    Comment = 0x30,
    UserJoin = 0x40,
    UserLeave = 0x41,
    Ping = 0xF0,
    Pong = 0xF1,
    Error = 0xFF,

    // Binary messages (audio data)
    AudioChunk = 0x80,
    AudioMetadata = 0x81,
    AudioAck = 0x82
};

/**
 * @brief Message frame header (network byte order)
 */
#pragma pack(push, 1)
struct MessageHeader {
    MessageType type;
    uint8_t flags;          // Bit 0: compressed, Bit 1: encrypted
    uint16_t payloadSize;   // For control messages
};

struct AudioChunkHeader {
    MessageHeader base;
    uint32_t sequenceNumber;
    uint32_t samplePosition;
    uint16_t channelCount;
    uint16_t sampleCount;
    // Followed by raw float32 PCM data
};
#pragma pack(pop)

/**
 * @brief Configuration for WebSocket connection
 */
struct WebSocketConfig {
    std::string host = "wss://collab.molinantro.app";
    uint16_t port = 443;
    bool useTLS = true;
    std::string path = "/ws/v1/session";

    // Timeouts (milliseconds)
    int connectTimeoutMs = 10000;
    int pingIntervalMs = 30000;
    int pongTimeoutMs = 10000;

    // Reconnection
    bool autoReconnect = true;
    int maxReconnectAttempts = 10;
    int initialReconnectDelayMs = 1000;
    int maxReconnectDelayMs = 30000;
    float reconnectBackoffMultiplier = 2.0f;

    // Buffer sizes
    size_t maxMessageSize = 16 * 1024 * 1024;  // 16MB for audio
};

/**
 * @brief Callbacks for WebSocket events
 */
struct WebSocketCallbacks {
    std::function<void(ConnectionState)> onStateChange;
    std::function<void(MessageType, const std::string&)> onTextMessage;
    std::function<void(MessageType, const std::vector<uint8_t>&)> onBinaryMessage;
    std::function<void(const std::string&)> onError;
    std::function<void()> onConnected;
    std::function<void()> onDisconnected;
};

// WebSocket++ type aliases
using WsClient = websocketpp::client<websocketpp::config::asio_tls_client>;
using WsConnectionPtr = websocketpp::connection_hdl;
using WsMessage = websocketpp::config::asio_client::message_type::ptr;
using SslContext = websocketpp::lib::shared_ptr<asio::ssl::context>;

/**
 * @brief Production WebSocket client with real networking
 *
 * Thread-safe implementation using websocketpp library.
 * Supports both text (JSON) and binary (audio) messages.
 */
class WebSocketClient {
public:
    WebSocketClient();
    ~WebSocketClient();

    // Non-copyable, non-movable
    WebSocketClient(const WebSocketClient&) = delete;
    WebSocketClient& operator=(const WebSocketClient&) = delete;

    /**
     * @brief Configure the client
     */
    void configure(const WebSocketConfig& config);

    /**
     * @brief Set event callbacks
     */
    void setCallbacks(WebSocketCallbacks callbacks);

    /**
     * @brief Connect to server with authentication token
     * @param serverUri Full WebSocket URI (e.g., wss://server.com/path)
     * @param authToken JWT authentication token
     * @return true if connection initiated successfully
     */
    bool connect(const std::string& serverUri, const std::string& authToken);

    /**
     * @brief Disconnect gracefully
     */
    void disconnect();

    /**
     * @brief Send text message (JSON)
     * @param type Message type code
     * @param jsonPayload JSON string payload
     */
    void sendTextMessage(MessageType type, const std::string& jsonPayload);

    /**
     * @brief Send binary message (audio chunk)
     * @param type Message type code
     * @param data Binary data
     */
    void sendBinaryMessage(MessageType type, const std::vector<uint8_t>& data);

    /**
     * @brief Send raw audio chunk
     * @param sequenceNumber Sequence number for ordering
     * @param samplePosition Sample position in timeline
     * @param data Audio samples (interleaved float32)
     * @param channels Number of channels
     * @param samples Number of samples per channel
     */
    void sendAudioChunk(uint32_t sequenceNumber, uint32_t samplePosition,
                        const float* data, int channels, int samples);

    /**
     * @brief Get current connection state
     */
    ConnectionState getState() const { return state_.load(); }

    /**
     * @brief Check if connected
     */
    bool isConnected() const { return state_.load() == ConnectionState::Connected; }

    /**
     * @brief Get latency estimate (round-trip time in ms)
     */
    double getLatencyMs() const { return latencyMs_.load(); }

    /**
     * @brief Get statistics
     */
    struct Stats {
        uint64_t messagesSent = 0;
        uint64_t messagesReceived = 0;
        uint64_t bytesSent = 0;
        uint64_t bytesReceived = 0;
        uint64_t reconnectCount = 0;
        uint64_t errorCount = 0;
    };
    Stats getStats() const;

private:
    // WebSocket++ event handlers
    void onOpen(WsConnectionPtr hdl);
    void onClose(WsConnectionPtr hdl);
    void onFail(WsConnectionPtr hdl);
    void onMessage(WsConnectionPtr hdl, WsMessage msg);
    SslContext onTlsInit(WsConnectionPtr hdl);

    // Internal helpers
    void setState(ConnectionState newState);
    void scheduleReconnect();
    void reconnectLoop();
    void processOutgoingQueue();
    void runIoService();

    // Parse incoming message
    void parseMessage(const std::string& payload, bool isBinary);

    // WebSocket++ client
    std::unique_ptr<WsClient> client_;
    WsConnectionPtr connectionHdl_;
    std::atomic<bool> hasConnection_{false};  // [SECURITY FIX] Atomic for thread safety

    // Configuration
    WebSocketConfig config_;
    WebSocketCallbacks callbacks_;
    std::string serverUri_;
    std::string authToken_;

    // State
    std::atomic<ConnectionState> state_{ConnectionState::Disconnected};
    std::atomic<double> latencyMs_{0.0};
    std::atomic<int> reconnectAttempts_{0};

    // Ping/pong timing
    std::chrono::steady_clock::time_point lastPingSent_;
    std::chrono::steady_clock::time_point lastPongReceived_;

    // Message queues (thread-safe)
    struct OutgoingMessage {
        bool isBinary;
        std::string data;
    };
    std::queue<OutgoingMessage> outgoingQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCv_;

    // Threading
    std::thread ioThread_;
    std::thread sendThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> shouldReconnect_{false};

    // Statistics
    mutable std::mutex statsMutex_;
    Stats stats_;

    // Random for jitter
    std::mt19937 rng_;
};

// [SECURITY] Maximum payload size for control messages
constexpr size_t MAX_CONTROL_PAYLOAD_SIZE = 65535;
// [SECURITY] Maximum queue size to prevent DoS
constexpr size_t MAX_OUTGOING_QUEUE_SIZE = 1000;
constexpr size_t MAX_OUTGOING_QUEUE_BYTES = 50 * 1024 * 1024;  // 50MB

/**
 * @brief Serialize message with header
 * @throws std::runtime_error if payload exceeds MAX_CONTROL_PAYLOAD_SIZE
 */
inline std::string serializeMessage(MessageType type, const std::string& payload) {
    // [SECURITY FIX] Validate payload size to prevent integer overflow
    if (payload.size() > MAX_CONTROL_PAYLOAD_SIZE) {
        throw std::runtime_error("Payload size exceeds maximum allowed: " +
                                 std::to_string(payload.size()) + " > " +
                                 std::to_string(MAX_CONTROL_PAYLOAD_SIZE));
    }

    MessageHeader header;
    header.type = type;
    header.flags = 0;
    header.payloadSize = static_cast<uint16_t>(payload.size());

    std::string result;
    result.resize(sizeof(MessageHeader) + payload.size());
    std::memcpy(result.data(), &header, sizeof(MessageHeader));
    std::memcpy(result.data() + sizeof(MessageHeader), payload.data(), payload.size());

    return result;
}

/**
 * @brief Deserialize message header
 */
inline bool deserializeHeader(const std::string& data, MessageHeader& header) {
    if (data.size() < sizeof(MessageHeader)) {
        return false;
    }
    std::memcpy(&header, data.data(), sizeof(MessageHeader));
    return true;
}

} // namespace Net
} // namespace Cloud
} // namespace MolinAntro
