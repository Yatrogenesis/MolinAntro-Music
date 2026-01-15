/**
 * @file WebSocketClient.cpp
 * @brief REAL WebSocket client implementation - NO SIMULATIONS
 *
 * This file contains ACTUAL networking code that:
 * - Opens real TCP sockets
 * - Performs TLS handshakes
 * - Sends/receives WebSocket frames
 * - Handles reconnection with exponential backoff
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "cloud/WebSocketClient.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace MolinAntro {
namespace Cloud {
namespace Net {

//=============================================================================
// Constructor / Destructor
//=============================================================================

WebSocketClient::WebSocketClient()
    : client_(std::make_unique<WsClient>())
    , rng_(std::random_device{}())
{
    // Initialize websocketpp client
    client_->clear_access_channels(websocketpp::log::alevel::all);
    client_->clear_error_channels(websocketpp::log::elevel::all);

    // Enable only important logs
    client_->set_access_channels(websocketpp::log::alevel::connect);
    client_->set_access_channels(websocketpp::log::alevel::disconnect);
    client_->set_error_channels(websocketpp::log::elevel::rerror);
    client_->set_error_channels(websocketpp::log::elevel::fatal);

    // Initialize ASIO
    client_->init_asio();

    // Set handlers
    client_->set_open_handler([this](WsConnectionPtr hdl) {
        onOpen(hdl);
    });

    client_->set_close_handler([this](WsConnectionPtr hdl) {
        onClose(hdl);
    });

    client_->set_fail_handler([this](WsConnectionPtr hdl) {
        onFail(hdl);
    });

    client_->set_message_handler([this](WsConnectionPtr hdl, WsMessage msg) {
        onMessage(hdl, msg);
    });

    client_->set_tls_init_handler([this](WsConnectionPtr hdl) {
        return onTlsInit(hdl);
    });
}

WebSocketClient::~WebSocketClient() {
    disconnect();
}

//=============================================================================
// Configuration
//=============================================================================

void WebSocketClient::configure(const WebSocketConfig& config) {
    config_ = config;

    // Set max message size
    client_->set_max_message_size(config_.maxMessageSize);
}

void WebSocketClient::setCallbacks(WebSocketCallbacks callbacks) {
    callbacks_ = std::move(callbacks);
}

//=============================================================================
// Connection Management
//=============================================================================

bool WebSocketClient::connect(const std::string& serverUri, const std::string& authToken) {
    if (state_.load() == ConnectionState::Connected ||
        state_.load() == ConnectionState::Connecting) {
        return false;
    }

    serverUri_ = serverUri;
    authToken_ = authToken;
    running_ = true;
    reconnectAttempts_ = 0;

    setState(ConnectionState::Connecting);

    try {
        websocketpp::lib::error_code ec;

        // Create connection
        WsClient::connection_ptr con = client_->get_connection(serverUri_, ec);

        if (ec) {
            std::string errMsg = "Connection creation failed: " + ec.message();
            if (callbacks_.onError) {
                callbacks_.onError(errMsg);
            }
            setState(ConnectionState::Failed);
            return false;
        }

        // Set connection timeout
        con->set_open_handshake_timeout(config_.connectTimeoutMs);

        // Add authentication header
        con->append_header("Authorization", "Bearer " + authToken_);
        con->append_header("X-Client-Version", "2.0.0");
        con->append_header("X-Protocol-Version", "1");

        // Store connection handle
        connectionHdl_ = con->get_handle();
        hasConnection_ = true;

        // Queue the connection
        client_->connect(con);

        // Start I/O thread if not running
        if (!ioThread_.joinable()) {
            ioThread_ = std::thread(&WebSocketClient::runIoService, this);
        }

        // Start send thread
        if (!sendThread_.joinable()) {
            sendThread_ = std::thread(&WebSocketClient::processOutgoingQueue, this);
        }

        return true;

    } catch (const std::exception& e) {
        std::string errMsg = "Connection exception: ";
        errMsg += e.what();
        if (callbacks_.onError) {
            callbacks_.onError(errMsg);
        }
        setState(ConnectionState::Failed);
        return false;
    }
}

void WebSocketClient::disconnect() {
    running_ = false;
    shouldReconnect_ = false;

    // Close WebSocket connection
    if (hasConnection_ && state_.load() == ConnectionState::Connected) {
        try {
            websocketpp::lib::error_code ec;
            client_->close(connectionHdl_, websocketpp::close::status::normal, "Client disconnect", ec);
        } catch (...) {
            // Ignore errors during disconnect
        }
    }

    hasConnection_ = false;

    // Wake up send thread
    queueCv_.notify_all();

    // Stop I/O service
    client_->stop();

    // Join threads
    if (ioThread_.joinable()) {
        ioThread_.join();
    }
    if (sendThread_.joinable()) {
        sendThread_.join();
    }

    setState(ConnectionState::Disconnected);
}

//=============================================================================
// Message Sending
//=============================================================================

void WebSocketClient::sendTextMessage(MessageType type, const std::string& jsonPayload) {
    if (state_.load() != ConnectionState::Connected) {
        return;
    }

    // Serialize with header
    std::string message;
    try {
        message = serializeMessage(type, jsonPayload);
    } catch (const std::runtime_error& e) {
        if (callbacks_.onError) {
            callbacks_.onError(std::string("Serialization error: ") + e.what());
        }
        return;
    }

    // [SECURITY FIX] Check queue limits before adding
    {
        std::lock_guard<std::mutex> lock(queueMutex_);

        if (outgoingQueue_.size() >= MAX_OUTGOING_QUEUE_SIZE) {
            if (callbacks_.onError) {
                callbacks_.onError("Outgoing queue full - message dropped");
            }
            return;
        }

        outgoingQueue_.push({false, message});
    }
    queueCv_.notify_one();
}

void WebSocketClient::sendBinaryMessage(MessageType type, const std::vector<uint8_t>& data) {
    if (state_.load() != ConnectionState::Connected) {
        return;
    }

    // [SECURITY FIX] Validate data size
    if (data.size() > MAX_CONTROL_PAYLOAD_SIZE) {
        if (callbacks_.onError) {
            callbacks_.onError("Binary message too large");
        }
        return;
    }

    // Create message with header
    std::string message;
    message.resize(sizeof(MessageHeader) + data.size());

    MessageHeader header;
    header.type = type;
    header.flags = 0;
    header.payloadSize = static_cast<uint16_t>(data.size());

    std::memcpy(message.data(), &header, sizeof(MessageHeader));
    std::memcpy(message.data() + sizeof(MessageHeader), data.data(), data.size());

    // [SECURITY FIX] Check queue limits before adding
    {
        std::lock_guard<std::mutex> lock(queueMutex_);

        if (outgoingQueue_.size() >= MAX_OUTGOING_QUEUE_SIZE) {
            if (callbacks_.onError) {
                callbacks_.onError("Outgoing queue full - message dropped");
            }
            return;
        }

        outgoingQueue_.push({true, message});
    }
    queueCv_.notify_one();
}

void WebSocketClient::sendAudioChunk(uint32_t sequenceNumber, uint32_t samplePosition,
                                      const float* data, int channels, int samples) {
    if (state_.load() != ConnectionState::Connected || !data) {
        return;
    }

    // Calculate total size
    size_t audioDataSize = channels * samples * sizeof(float);
    size_t totalSize = sizeof(AudioChunkHeader) + audioDataSize;

    // Create binary message
    std::string message;
    message.resize(totalSize);

    // Fill header
    AudioChunkHeader header;
    header.base.type = MessageType::AudioChunk;
    header.base.flags = 0;
    header.base.payloadSize = 0;  // Not used for audio
    header.sequenceNumber = sequenceNumber;
    header.samplePosition = samplePosition;
    header.channelCount = static_cast<uint16_t>(channels);
    header.sampleCount = static_cast<uint16_t>(samples);

    std::memcpy(message.data(), &header, sizeof(AudioChunkHeader));
    std::memcpy(message.data() + sizeof(AudioChunkHeader), data, audioDataSize);

    // Queue for sending (binary)
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        outgoingQueue_.push({true, message});
    }
    queueCv_.notify_one();

    // Update stats
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.bytesSent += totalSize;
        stats_.messagesSent++;
    }
}

//=============================================================================
// WebSocket++ Event Handlers
//=============================================================================

void WebSocketClient::onOpen(WsConnectionPtr hdl) {
    connectionHdl_ = hdl;
    hasConnection_ = true;
    reconnectAttempts_ = 0;

    setState(ConnectionState::Connected);

    // Send ping timestamp for latency measurement
    lastPingSent_ = std::chrono::steady_clock::now();

    if (callbacks_.onConnected) {
        callbacks_.onConnected();
    }

    // Send authentication message
    std::stringstream ss;
    ss << R"({"token":")" << authToken_ << R"(","version":"2.0.0"})";
    sendTextMessage(MessageType::Authenticate, ss.str());
}

void WebSocketClient::onClose(WsConnectionPtr hdl) {
    hasConnection_ = false;

    ConnectionState prevState = state_.load();
    setState(ConnectionState::Disconnected);

    if (callbacks_.onDisconnected) {
        callbacks_.onDisconnected();
    }

    // Auto-reconnect if enabled and was previously connected
    if (config_.autoReconnect && running_ &&
        prevState == ConnectionState::Connected) {
        scheduleReconnect();
    }
}

void WebSocketClient::onFail(WsConnectionPtr hdl) {
    hasConnection_ = false;

    WsClient::connection_ptr con = client_->get_con_from_hdl(hdl);
    std::string errMsg = "Connection failed: " + con->get_ec().message();

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.errorCount++;
    }

    if (callbacks_.onError) {
        callbacks_.onError(errMsg);
    }

    // Auto-reconnect
    if (config_.autoReconnect && running_) {
        scheduleReconnect();
    } else {
        setState(ConnectionState::Failed);
    }
}

void WebSocketClient::onMessage(WsConnectionPtr hdl, WsMessage msg) {
    // Update stats
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.messagesReceived++;
        stats_.bytesReceived += msg->get_payload().size();
    }

    // Parse message
    bool isBinary = (msg->get_opcode() == websocketpp::frame::opcode::binary);
    parseMessage(msg->get_payload(), isBinary);

    // Update latency on pong
    if (!isBinary && msg->get_payload().size() >= sizeof(MessageHeader)) {
        MessageHeader header;
        std::memcpy(&header, msg->get_payload().data(), sizeof(MessageHeader));

        if (header.type == MessageType::Pong) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - lastPingSent_).count();
            latencyMs_ = static_cast<double>(elapsed);
            lastPongReceived_ = now;
        }
    }
}

SslContext WebSocketClient::onTlsInit(WsConnectionPtr hdl) {
    SslContext ctx = websocketpp::lib::make_shared<asio::ssl::context>(
        asio::ssl::context::tlsv12_client);

    try {
        ctx->set_options(
            asio::ssl::context::default_workarounds |
            asio::ssl::context::no_sslv2 |
            asio::ssl::context::no_sslv3 |
            asio::ssl::context::no_tlsv1 |
            asio::ssl::context::no_tlsv1_1 |
            asio::ssl::context::single_dh_use
        );

        // Use system certificate store
        ctx->set_default_verify_paths();

        // Enable certificate verification with hostname check
        ctx->set_verify_mode(asio::ssl::verify_peer);

        // [SECURITY FIX] Add hostname verification callback
        ctx->set_verify_callback([this](bool preverified, asio::ssl::verify_context& vctx) -> bool {
            // If basic chain verification failed, reject immediately
            if (!preverified) {
                return false;
            }

            // Extract hostname from serverUri_
            std::string expectedHost = "collab.molinantro.app";  // Default
            if (!serverUri_.empty()) {
                // Parse host from wss://host/path
                size_t start = serverUri_.find("://");
                if (start != std::string::npos) {
                    start += 3;
                    size_t end = serverUri_.find('/', start);
                    if (end == std::string::npos) end = serverUri_.find(':', start);
                    if (end == std::string::npos) end = serverUri_.length();
                    expectedHost = serverUri_.substr(start, end - start);
                }
            }

            // Get certificate from context
            X509* cert = X509_STORE_CTX_get_current_cert(vctx.native_handle());
            if (!cert) {
                return false;
            }

            // Check Common Name (CN) or Subject Alternative Name (SAN)
            char commonName[256] = {0};
            X509_NAME* subject = X509_get_subject_name(cert);
            if (subject) {
                X509_NAME_get_text_by_NID(subject, NID_commonName, commonName, sizeof(commonName) - 1);
            }

            if (expectedHost == commonName) {
                return true;
            }

            // Also check Subject Alternative Names
            GENERAL_NAMES* sans = static_cast<GENERAL_NAMES*>(
                X509_get_ext_d2i(cert, NID_subject_alt_name, nullptr, nullptr));
            if (sans) {
                for (int i = 0; i < sk_GENERAL_NAME_num(sans); ++i) {
                    GENERAL_NAME* gen = sk_GENERAL_NAME_value(sans, i);
                    if (gen->type == GEN_DNS) {
                        const char* dns = reinterpret_cast<const char*>(
                            ASN1_STRING_get0_data(gen->d.dNSName));
                        if (dns && expectedHost == dns) {
                            GENERAL_NAMES_free(sans);
                            return true;
                        }
                    }
                }
                GENERAL_NAMES_free(sans);
            }

            // Hostname mismatch - potential MITM attack
            std::cerr << "[SECURITY] TLS hostname mismatch: expected " << expectedHost
                      << ", got " << commonName << std::endl;
            return false;
        });

    } catch (const std::exception& e) {
        // Log but continue - some systems may not have certs
        std::cerr << "TLS init warning: " << e.what() << std::endl;
    }

    return ctx;
}

//=============================================================================
// Internal Helpers
//=============================================================================

void WebSocketClient::setState(ConnectionState newState) {
    ConnectionState oldState = state_.exchange(newState);

    if (oldState != newState && callbacks_.onStateChange) {
        callbacks_.onStateChange(newState);
    }
}

void WebSocketClient::scheduleReconnect() {
    int attempts = reconnectAttempts_.load();

    if (attempts >= config_.maxReconnectAttempts) {
        setState(ConnectionState::Failed);
        if (callbacks_.onError) {
            callbacks_.onError("Max reconnection attempts reached");
        }
        return;
    }

    setState(ConnectionState::Reconnecting);

    // Calculate delay with exponential backoff + jitter
    double baseDelay = config_.initialReconnectDelayMs;
    double maxDelay = config_.maxReconnectDelayMs;
    double delay = std::min(
        baseDelay * std::pow(config_.reconnectBackoffMultiplier, attempts),
        maxDelay
    );

    // Add 0-20% jitter
    std::uniform_real_distribution<> jitterDist(0.0, 0.2);
    delay *= (1.0 + jitterDist(rng_));

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.reconnectCount++;
    }

    // Schedule reconnect
    shouldReconnect_ = true;
    reconnectAttempts_++;

    // Spawn reconnect thread
    std::thread([this, delay]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay)));

        if (shouldReconnect_ && running_) {
            // Reset client for new connection
            client_->reset();

            // Attempt reconnect
            websocketpp::lib::error_code ec;
            WsClient::connection_ptr con = client_->get_connection(serverUri_, ec);

            if (!ec) {
                con->set_open_handshake_timeout(config_.connectTimeoutMs);
                con->append_header("Authorization", "Bearer " + authToken_);
                connectionHdl_ = con->get_handle();
                client_->connect(con);
            }
        }
    }).detach();
}

void WebSocketClient::processOutgoingQueue() {
    while (running_) {
        OutgoingMessage msg;
        bool hasMessage = false;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCv_.wait(lock, [this] {
                return !outgoingQueue_.empty() || !running_;
            });

            if (!running_) break;

            if (!outgoingQueue_.empty()) {
                msg = std::move(outgoingQueue_.front());
                outgoingQueue_.pop();
                hasMessage = true;
            }
        }

        if (hasMessage && hasConnection_ && state_.load() == ConnectionState::Connected) {
            try {
                websocketpp::lib::error_code ec;

                if (msg.isBinary) {
                    client_->send(connectionHdl_, msg.data,
                                  websocketpp::frame::opcode::binary, ec);
                } else {
                    client_->send(connectionHdl_, msg.data,
                                  websocketpp::frame::opcode::text, ec);
                }

                if (ec) {
                    std::lock_guard<std::mutex> lock(statsMutex_);
                    stats_.errorCount++;
                } else {
                    std::lock_guard<std::mutex> lock(statsMutex_);
                    stats_.bytesSent += msg.data.size();
                    stats_.messagesSent++;
                }

            } catch (const std::exception& e) {
                if (callbacks_.onError) {
                    callbacks_.onError(std::string("Send error: ") + e.what());
                }
            }
        }
    }
}

void WebSocketClient::runIoService() {
    try {
        client_->run();
    } catch (const std::exception& e) {
        if (callbacks_.onError) {
            callbacks_.onError(std::string("I/O error: ") + e.what());
        }
    }
}

void WebSocketClient::parseMessage(const std::string& payload, bool isBinary) {
    // [SECURITY FIX] Validate minimum header size
    if (payload.size() < sizeof(MessageHeader)) {
        if (callbacks_.onError) {
            callbacks_.onError("Received malformed message: too short for header");
        }
        return;
    }

    MessageHeader header;
    std::memcpy(&header, payload.data(), sizeof(MessageHeader));

    // [SECURITY FIX] Validate payload size matches header
    size_t expectedTotalSize = sizeof(MessageHeader) + header.payloadSize;
    if (payload.size() < expectedTotalSize) {
        if (callbacks_.onError) {
            callbacks_.onError("Received malformed message: payload size mismatch");
        }
        return;
    }

    // [SECURITY FIX] Extract only the declared payload size (not the rest)
    std::string messagePayload = payload.substr(sizeof(MessageHeader), header.payloadSize);

    // [SECURITY FIX] For binary audio messages, validate audio header
    if (isBinary && header.type == MessageType::AudioChunk) {
        if (messagePayload.size() < sizeof(AudioChunkHeader) - sizeof(MessageHeader)) {
            if (callbacks_.onError) {
                callbacks_.onError("Received malformed audio chunk: too short");
            }
            return;
        }

        // Check for integer overflow in audio size calculation
        AudioChunkHeader audioHeader;
        std::memcpy(&audioHeader, payload.data(), sizeof(AudioChunkHeader));

        size_t audioDataSize = static_cast<size_t>(audioHeader.channelCount) *
                               static_cast<size_t>(audioHeader.sampleCount) * sizeof(float);

        // [SECURITY FIX] Validate audio dimensions against actual payload
        size_t expectedAudioSize = sizeof(AudioChunkHeader) + audioDataSize;
        if (payload.size() < expectedAudioSize || audioDataSize > config_.maxMessageSize) {
            if (callbacks_.onError) {
                callbacks_.onError("Received malformed audio chunk: size overflow");
            }
            return;
        }
    }

    if (isBinary) {
        if (callbacks_.onBinaryMessage) {
            std::vector<uint8_t> data(messagePayload.begin(), messagePayload.end());
            callbacks_.onBinaryMessage(header.type, data);
        }
    } else {
        if (callbacks_.onTextMessage) {
            callbacks_.onTextMessage(header.type, messagePayload);
        }
    }
}

WebSocketClient::Stats WebSocketClient::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

} // namespace Net
} // namespace Cloud
} // namespace MolinAntro
