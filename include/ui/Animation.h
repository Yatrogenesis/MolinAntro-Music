#pragma once

#include "ui/Theme.h"
#include <functional>
#include <vector>
#include <memory>
#include <chrono>

namespace MolinAntro {
namespace UI {

/**
 * @brief Easing functions for smooth animations
 */
class Easing {
public:
    using Function = std::function<float(float)>;

    // Standard easing functions
    static float linear(float t);
    static float easeInQuad(float t);
    static float easeOutQuad(float t);
    static float easeInOutQuad(float t);
    static float easeInCubic(float t);
    static float easeOutCubic(float t);
    static float easeInOutCubic(float t);
    static float easeInQuart(float t);
    static float easeOutQuart(float t);
    static float easeInOutQuart(float t);
    static float easeInExpo(float t);
    static float easeOutExpo(float t);
    static float easeInOutExpo(float t);
    static float easeInCirc(float t);
    static float easeOutCirc(float t);
    static float easeInOutCirc(float t);
    static float easeInBack(float t);
    static float easeOutBack(float t);
    static float easeInOutBack(float t);
    static float easeInElastic(float t);
    static float easeOutElastic(float t);
    static float easeInOutElastic(float t);
    static float easeInBounce(float t);
    static float easeOutBounce(float t);
    static float easeInOutBounce(float t);

    // Get function by Animation::Easing enum
    static Function getFunction(Animation::Easing type);
};

/**
 * @brief Animated value interpolation
 */
template<typename T>
class AnimatedValue {
public:
    AnimatedValue(const T& initialValue = T())
        : currentValue_(initialValue)
        , targetValue_(initialValue)
        , startValue_(initialValue) {}

    // Set target value and animate to it
    void animateTo(const T& target, float duration = 0.2f,
                   Animation::Easing easing = Animation::Easing::EaseOut);

    // Set value immediately (no animation)
    void setValue(const T& value) {
        currentValue_ = value;
        targetValue_ = value;
        startValue_ = value;
        isAnimating_ = false;
    }

    // Get current value
    const T& getValue() const { return currentValue_; }
    const T& getTargetValue() const { return targetValue_; }

    // Check if animating
    bool isAnimating() const { return isAnimating_; }

    // Update (call each frame)
    void update(float deltaTime);

    // Callbacks
    using CompletionCallback = std::function<void()>;
    void onComplete(CompletionCallback callback) { completionCallback_ = callback; }

private:
    T currentValue_;
    T targetValue_;
    T startValue_;
    float duration_{0.2f};
    float elapsed_{0.0f};
    bool isAnimating_{false};
    Animation::Easing easingType_{Animation::Easing::EaseOut};
    CompletionCallback completionCallback_;

    T interpolate(const T& start, const T& end, float t);
};

/**
 * @brief Timeline-based animation system
 */
class Animator {
public:
    Animator();
    ~Animator();

    enum class State {
        Stopped,
        Playing,
        Paused
    };

    // Playback control
    void play();
    void pause();
    void stop();
    void setTime(float time);
    float getTime() const { return time_; }
    State getState() const { return state_; }

    // Duration
    void setDuration(float duration) { duration_ = duration; }
    float getDuration() const { return duration_; }

    // Loop
    void setLoop(bool loop) { loop_ = loop; }
    bool isLooping() const { return loop_; }

    // Speed
    void setSpeed(float speed) { speed_ = speed; }
    float getSpeed() const { return speed_; }

    // Keyframes
    struct Keyframe {
        float time;
        float value;
        Animation::Easing easing{Animation::Easing::Linear};
    };

    void addKeyframe(float time, float value,
                     Animation::Easing easing = Animation::Easing::Linear);
    void clearKeyframes();
    float getValueAtTime(float time) const;

    // Update
    void update(float deltaTime);

    // Callbacks
    using ValueCallback = std::function<void(float)>;
    using CompletionCallback = std::function<void()>;

    void onUpdate(ValueCallback callback) { updateCallback_ = callback; }
    void onComplete(CompletionCallback callback) { completeCallback_ = callback; }

private:
    std::vector<Keyframe> keyframes_;
    State state_{State::Stopped};
    float time_{0.0f};
    float duration_{1.0f};
    float speed_{1.0f};
    bool loop_{false};
    ValueCallback updateCallback_;
    CompletionCallback completeCallback_;

    void sortKeyframes();
};

/**
 * @brief Spring physics animation for natural motion
 */
class SpringAnimation {
public:
    SpringAnimation();

    // Physics properties
    void setStiffness(float stiffness) { stiffness_ = stiffness; }
    void setDamping(float damping) { damping_ = damping; }
    void setMass(float mass) { mass_ = mass; }

    // Target
    void setTarget(float target) { target_ = target; }
    float getTarget() const { return target_; }

    // Value
    void setValue(float value);
    float getValue() const { return value_; }
    float getVelocity() const { return velocity_; }

    // State
    bool isAtRest() const;

    // Update
    void update(float deltaTime);

private:
    float value_{0.0f};
    float velocity_{0.0f};
    float target_{0.0f};
    float stiffness_{300.0f};   ///< Spring stiffness
    float damping_{30.0f};      ///< Damping coefficient
    float mass_{1.0f};          ///< Mass
    float restThreshold_{0.001f}; ///< Threshold for "at rest"
};

/**
 * @brief GPU-accelerated animation manager
 */
class AnimationManager {
public:
    static AnimationManager& getInstance();

    // Animation registration
    using AnimationID = size_t;

    template<typename T>
    AnimationID animate(AnimatedValue<T>* value, const T& target,
                       float duration, Animation::Easing easing = Animation::Easing::EaseOut);

    void cancelAnimation(AnimationID id);
    void cancelAllAnimations();

    // Global control
    void setPaused(bool paused) { paused_ = paused; }
    bool isPaused() const { return paused_; }

    void setGlobalSpeed(float speed) { globalSpeed_ = speed; }
    float getGlobalSpeed() const { return globalSpeed_; }

    // GPU acceleration
    void setUseGPU(bool useGPU) { useGPU_ = useGPU; }
    bool isUsingGPU() const { return useGPU_; }

    // Update all animations (call once per frame)
    void update(float deltaTime);

    // Statistics
    size_t getActiveAnimationCount() const { return activeAnimations_.size(); }
    float getAverageFPS() const { return avgFPS_; }

private:
    AnimationManager();
    ~AnimationManager() = default;
    AnimationManager(const AnimationManager&) = delete;
    AnimationManager& operator=(const AnimationManager&) = delete;

    struct AnimationState {
        AnimationID id;
        std::function<void(float)> updateFunc;
        std::function<bool()> isActiveFunc;
    };

    std::vector<AnimationState> activeAnimations_;
    AnimationID nextID_{1};
    bool paused_{false};
    float globalSpeed_{1.0f};
    bool useGPU_{true};

    // Performance tracking
    std::chrono::steady_clock::time_point lastUpdate_;
    float avgFPS_{60.0f};
};

/**
 * @brief Transition effects between UI states
 */
class Transition {
public:
    enum class Type {
        Fade,
        Slide,
        Scale,
        Rotate,
        Blur,
        Dissolve
    };

    Transition(Type type, float duration = 0.3f);

    void setType(Type type) { type_ = type; }
    void setDuration(float duration) { duration_ = duration; }
    void setEasing(Animation::Easing easing) { easing_ = easing; }

    // Start transition
    void start();
    void reverse();
    void reset();

    // State
    bool isActive() const { return active_; }
    float getProgress() const { return progress_; }

    // Update
    void update(float deltaTime);

    // Callbacks
    using CompletionCallback = std::function<void()>;
    void onComplete(CompletionCallback callback) { completeCallback_ = callback; }

    // Get transformation matrix/values for rendering
    struct Transform {
        float opacity{1.0f};
        float translateX{0.0f};
        float translateY{0.0f};
        float scaleX{1.0f};
        float scaleY{1.0f};
        float rotation{0.0f};
        float blur{0.0f};
    };

    Transform getTransform() const;

private:
    Type type_;
    float duration_;
    Animation::Easing easing_;
    bool active_{false};
    bool reversed_{false};
    float progress_{0.0f};
    CompletionCallback completeCallback_;
};

/**
 * @brief Gesture recognizer for touch/mouse gestures
 */
class GestureRecognizer {
public:
    enum class GestureType {
        Tap,
        DoubleTap,
        LongPress,
        Pan,
        Pinch,
        Rotate,
        Swipe
    };

    GestureRecognizer();

    // Input
    void onMouseDown(float x, float y);
    void onMouseMove(float x, float y);
    void onMouseUp(float x, float y);
    void onWheel(float deltaX, float deltaY);

    // Multi-touch
    void addTouch(int id, float x, float y);
    void updateTouch(int id, float x, float y);
    void removeTouch(int id);

    // Callbacks
    using GestureCallback = std::function<void(GestureType, float, float)>;
    void onGesture(GestureCallback callback) { gestureCallback_ = callback; }

    // Configuration
    void setDoubleTapDelay(float delay) { doubleTapDelay_ = delay; }
    void setLongPressDelay(float delay) { longPressDelay_ = delay; }
    void setSwipeThreshold(float threshold) { swipeThreshold_ = threshold; }

private:
    struct Touch {
        int id;
        float x, y;
        float startX, startY;
        std::chrono::steady_clock::time_point startTime;
    };

    std::vector<Touch> touches_;
    GestureCallback gestureCallback_;

    // Timing
    std::chrono::steady_clock::time_point lastTapTime_;
    float doubleTapDelay_{0.3f};
    float longPressDelay_{0.5f};
    float swipeThreshold_{50.0f};

    // Detection
    bool detectTap();
    bool detectDoubleTap();
    bool detectLongPress();
    bool detectPan();
    bool detectPinch();
    bool detectRotate();
    bool detectSwipe();
};

} // namespace UI
} // namespace MolinAntro
