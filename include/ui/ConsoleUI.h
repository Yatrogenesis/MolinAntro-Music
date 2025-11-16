#pragma once

#include "core/AudioEngine.h"
#include "core/Transport.h"
#include <string>
#include <functional>

namespace MolinAntro {
namespace UI {

/**
 * @brief Simple console-based UI for testing and MVP
 */
class ConsoleUI {
public:
    ConsoleUI(Core::AudioEngine& engine, Core::Transport& transport);
    ~ConsoleUI() = default;

    /**
     * @brief Run the console UI loop
     */
    void run();

    /**
     * @brief Process a single command
     * @return true if should continue, false to exit
     */
    bool processCommand(const std::string& command);

    /**
     * @brief Print help message
     */
    void printHelp() const;

    /**
     * @brief Print current status
     */
    void printStatus() const;

private:
    Core::AudioEngine& engine_;
    Core::Transport& transport_;

    void handlePlay();
    void handleStop();
    void handlePause();
    void handleRecord();
    void handleSetBPM(double bpm);
    void handleExit();
};

} // namespace UI
} // namespace MolinAntro
