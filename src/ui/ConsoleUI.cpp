#include "ui/ConsoleUI.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace MolinAntro {
namespace UI {

ConsoleUI::ConsoleUI(Core::AudioEngine& engine, Core::Transport& transport)
    : engine_(engine), transport_(transport) {
}

void ConsoleUI::run() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║         MolinAntro DAW - Console Interface           ║\n";
    std::cout << "║              Professional Audio Workstation           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    printHelp();

    std::string command;
    while (true) {
        std::cout << "\nMAW> ";
        std::getline(std::cin, command);

        if (!processCommand(command)) {
            break;
        }
    }

    std::cout << "Goodbye!\n";
}

bool ConsoleUI::processCommand(const std::string& command) {
    std::istringstream iss(command);
    std::string cmd;
    iss >> cmd;

    if (cmd == "play") {
        handlePlay();
    } else if (cmd == "stop") {
        handleStop();
    } else if (cmd == "pause") {
        handlePause();
    } else if (cmd == "record") {
        handleRecord();
    } else if (cmd == "bpm") {
        double bpm;
        if (iss >> bpm) {
            handleSetBPM(bpm);
        } else {
            std::cout << "Usage: bpm <value>\n";
        }
    } else if (cmd == "status") {
        printStatus();
    } else if (cmd == "help") {
        printHelp();
    } else if (cmd == "exit" || cmd == "quit") {
        return false;
    } else if (!cmd.empty()) {
        std::cout << "Unknown command: " << cmd << "\n";
        std::cout << "Type 'help' for available commands\n";
    }

    return true;
}

void ConsoleUI::printHelp() const {
    std::cout << "Available commands:\n";
    std::cout << "  play      - Start playback\n";
    std::cout << "  stop      - Stop playback\n";
    std::cout << "  pause     - Pause playback\n";
    std::cout << "  record    - Start recording\n";
    std::cout << "  bpm <n>   - Set tempo (BPM)\n";
    std::cout << "  status    - Show current status\n";
    std::cout << "  help      - Show this help\n";
    std::cout << "  exit/quit - Exit application\n";
}

void ConsoleUI::printStatus() const {
    auto timeInfo = transport_.getTimeInfo();

    std::cout << "\n╔═══ STATUS ═══════════════════════════════════════╗\n";
    std::cout << "║ Engine State: ";

    switch (engine_.getState()) {
        case Core::AudioEngine::State::Stopped:
            std::cout << "Stopped       ";
            break;
        case Core::AudioEngine::State::Playing:
            std::cout << "Playing       ";
            break;
        case Core::AudioEngine::State::Recording:
            std::cout << "Recording     ";
            break;
        case Core::AudioEngine::State::Paused:
            std::cout << "Paused        ";
            break;
    }
    std::cout << "                         ║\n";

    std::cout << "║ Sample Rate:  " << std::setw(6) << engine_.getSampleRate() << " Hz                         ║\n";
    std::cout << "║ Buffer Size:  " << std::setw(6) << engine_.getBufferSize() << " samples                     ║\n";
    std::cout << "║ CPU Usage:    " << std::fixed << std::setprecision(1)
              << std::setw(5) << engine_.getCPUUsage() << " %                           ║\n";
    std::cout << "║                                                   ║\n";
    std::cout << "║ BPM:          " << std::fixed << std::setprecision(2)
              << std::setw(6) << timeInfo.bpm << "                               ║\n";
    std::cout << "║ Time Sig:     " << timeInfo.numerator << "/" << timeInfo.denominator
              << "                                      ║\n";
    std::cout << "║ Position:     Bar " << std::setw(3) << timeInfo.bar
              << " | Beat " << timeInfo.beat << "                     ║\n";
    std::cout << "║ Time:         " << std::fixed << std::setprecision(2)
              << std::setw(8) << timeInfo.timeInSeconds << " sec                         ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n";
}

void ConsoleUI::handlePlay() {
    transport_.play();
    engine_.start();
    std::cout << "▶ Playing...\n";
}

void ConsoleUI::handleStop() {
    transport_.stop();
    engine_.stop();
    std::cout << "■ Stopped\n";
}

void ConsoleUI::handlePause() {
    transport_.pause();
    engine_.pause();
    std::cout << "❚❚ Paused\n";
}

void ConsoleUI::handleRecord() {
    transport_.record();
    engine_.start();
    std::cout << "● Recording...\n";
}

void ConsoleUI::handleSetBPM(double bpm) {
    transport_.setBPM(bpm);
    std::cout << "♪ BPM set to " << bpm << "\n";
}

void ConsoleUI::handleExit() {
    std::cout << "Shutting down...\n";
}

} // namespace UI
} // namespace MolinAntro
