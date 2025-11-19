// Animation.cpp - GPU-Accelerated Animation System
// MolinAntro DAW ACME Edition v3.0.0

#include "ui/Animation.h"
#include "ui/Preferences.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MolinAntro {
namespace UI {

// ============================================================================
// Easing Functions Implementation
// ============================================================================

float Easing::linear(float t) {
    return t;
}

float Easing::easeInQuad(float t) {
    return t * t;
}

float Easing::easeOutQuad(float t) {
    return t * (2.0f - t);
}

float Easing::easeInOutQuad(float t) {
    return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t;
}

float Easing::easeInCubic(float t) {
    return t * t * t;
}

float Easing::easeOutCubic(float t) {
    float t1 = t - 1.0f;
    return t1 * t1 * t1 + 1.0f;
}

float Easing::easeInOutCubic(float t) {
    return t < 0.5f ? 4.0f * t * t * t :
           (t - 1.0f) * (2.0f * t - 2.0f) * (2.0f * t - 2.0f) + 1.0f;
}

float Easing::easeInQuart(float t) {
    return t * t * t * t;
}

float Easing::easeOutQuart(float t) {
    float t1 = t - 1.0f;
    return 1.0f - t1 * t1 * t1 * t1;
}

float Easing::easeInOutQuart(float t) {
    float t1 = t - 1.0f;
    return t < 0.5f ? 8.0f * t * t * t * t :
           1.0f - 8.0f * t1 * t1 * t1 * t1;
}

float Easing::easeInExpo(float t) {
    return t == 0.0f ? 0.0f : std::pow(2.0f, 10.0f * (t - 1.0f));
}

float Easing::easeOutExpo(float t) {
    return t == 1.0f ? 1.0f : 1.0f - std::pow(2.0f, -10.0f * t);
}

float Easing::easeInOutExpo(float t) {
    if (t == 0.0f || t == 1.0f) return t;
    return t < 0.5f ? 0.5f * std::pow(2.0f, 20.0f * t - 10.0f) :
           0.5f * (2.0f - std::pow(2.0f, -20.0f * t + 10.0f));
}

float Easing::easeInCirc(float t) {
    return 1.0f - std::sqrt(1.0f - t * t);
}

float Easing::easeOutCirc(float t) {
    float t1 = t - 1.0f;
    return std::sqrt(1.0f - t1 * t1);
}

float Easing::easeInOutCirc(float t) {
    return t < 0.5f ? 0.5f * (1.0f - std::sqrt(1.0f - 4.0f * t * t)) :
           0.5f * (std::sqrt(1.0f - (2.0f * t - 2.0f) * (2.0f * t - 2.0f)) + 1.0f);
}

float Easing::easeInBack(float t) {
    const float c1 = 1.70158f;
    const float c3 = c1 + 1.0f;
    return c3 * t * t * t - c1 * t * t;
}

float Easing::easeOutBack(float t) {
    const float c1 = 1.70158f;
    const float c3 = c1 + 1.0f;
    float t1 = t - 1.0f;
    return 1.0f + c3 * t1 * t1 * t1 + c1 * t1 * t1;
}

float Easing::easeInOutBack(float t) {
    const float c1 = 1.70158f;
    const float c2 = c1 * 1.525f;

    return t < 0.5f
        ? 0.5f * ((2.0f * t) * (2.0f * t) * ((c2 + 1.0f) * 2.0f * t - c2))
        : 0.5f * ((2.0f * t - 2.0f) * (2.0f * t - 2.0f) *
                 ((c2 + 1.0f) * (2.0f * t - 2.0f) + c2) + 2.0f);
}

float Easing::easeInElastic(float t) {
    const float c4 = (2.0f * M_PI) / 3.0f;
    return t == 0.0f ? 0.0f : t == 1.0f ? 1.0f :
           -std::pow(2.0f, 10.0f * t - 10.0f) * std::sin((t * 10.0f - 10.75f) * c4);
}

float Easing::easeOutElastic(float t) {
    const float c4 = (2.0f * M_PI) / 3.0f;
    return t == 0.0f ? 0.0f : t == 1.0f ? 1.0f :
           std::pow(2.0f, -10.0f * t) * std::sin((t * 10.0f - 0.75f) * c4) + 1.0f;
}

float Easing::easeInOutElastic(float t) {
    const float c5 = (2.0f * M_PI) / 4.5f;
    return t == 0.0f ? 0.0f : t == 1.0f ? 1.0f : t < 0.5f
        ? -(std::pow(2.0f, 20.0f * t - 10.0f) * std::sin((20.0f * t - 11.125f) * c5)) * 0.5f
        : (std::pow(2.0f, -20.0f * t + 10.0f) * std::sin((20.0f * t - 11.125f) * c5)) * 0.5f + 1.0f;
}

float Easing::easeOutBounce(float t) {
    const float n1 = 7.5625f;
    const float d1 = 2.75f;

    if (t < 1.0f / d1) {
        return n1 * t * t;
    } else if (t < 2.0f / d1) {
        t -= 1.5f / d1;
        return n1 * t * t + 0.75f;
    } else if (t < 2.5f / d1) {
        t -= 2.25f / d1;
        return n1 * t * t + 0.9375f;
    } else {
        t -= 2.625f / d1;
        return n1 * t * t + 0.984375f;
    }
}

float Easing::easeInBounce(float t) {
    return 1.0f - easeOutBounce(1.0f - t);
}

float Easing::easeInOutBounce(float t) {
    return t < 0.5f
        ? (1.0f - easeOutBounce(1.0f - 2.0f * t)) * 0.5f
        : (1.0f + easeOutBounce(2.0f * t - 1.0f)) * 0.5f;
}

Easing::Function Easing::getFunction(Animation::Easing type) {
    switch (type) {
        case Animation::Easing::Linear:      return linear;
        case Animation::Easing::EaseIn:      return easeInCubic;
        case Animation::Easing::EaseOut:     return easeOutCubic;
        case Animation::Easing::EaseInOut:   return easeInOutCubic;
        case Animation::Easing::Bounce:      return easeOutBounce;
        case Animation::Easing::Elastic:     return easeOutElastic;
        case Animation::Easing::Back:        return easeOutBack;
        default:                             return linear;
    }
}

// ============================================================================
// Animator Implementation
// ============================================================================

Animator::Animator() {
}

Animator::~Animator() {
}

void Animator::play() {
    state_ = State::Playing;
}

void Animator::pause() {
    state_ = State::Paused;
}

void Animator::stop() {
    state_ = State::Stopped;
    time_ = 0.0f;
}

void Animator::setTime(float time) {
    time_ = std::clamp(time, 0.0f, duration_);
}

void Animator::addKeyframe(float time, float value, Animation::Easing easing) {
    keyframes_.push_back({time, value, easing});
    sortKeyframes();
}

void Animator::clearKeyframes() {
    keyframes_.clear();
}

float Animator::getValueAtTime(float time) const {
    if (keyframes_.empty()) return 0.0f;
    if (keyframes_.size() == 1) return keyframes_[0].value;

    // Find surrounding keyframes
    const Keyframe* prev = nullptr;
    const Keyframe* next = nullptr;

    for (size_t i = 0; i < keyframes_.size(); ++i) {
        if (keyframes_[i].time <= time) {
            prev = &keyframes_[i];
        }
        if (keyframes_[i].time >= time && !next) {
            next = &keyframes_[i];
            break;
        }
    }

    if (!prev) return keyframes_.front().value;
    if (!next) return keyframes_.back().value;
    if (prev == next) return prev->value;

    // Interpolate between keyframes
    float t = (time - prev->time) / (next->time - prev->time);
    auto easingFunc = Easing::getFunction(prev->easing);
    float easedT = easingFunc(t);

    return prev->value + (next->value - prev->value) * easedT;
}

void Animator::update(float deltaTime) {
    if (state_ != State::Playing) return;

    time_ += deltaTime * speed_;

    if (time_ >= duration_) {
        if (loop_) {
            time_ = std::fmod(time_, duration_);
        } else {
            time_ = duration_;
            state_ = State::Stopped;
            if (completeCallback_) {
                completeCallback_();
            }
        }
    }

    if (updateCallback_) {
        updateCallback_(getValueAtTime(time_));
    }
}

void Animator::sortKeyframes() {
    std::sort(keyframes_.begin(), keyframes_.end(),
              [](const Keyframe& a, const Keyframe& b) {
                  return a.time < b.time;
              });
}

// ============================================================================
// SpringAnimation Implementation
// ============================================================================

SpringAnimation::SpringAnimation() {
}

void SpringAnimation::setValue(float value) {
    value_ = value;
    velocity_ = 0.0f;
}

bool SpringAnimation::isAtRest() const {
    return std::abs(value_ - target_) < restThreshold_ &&
           std::abs(velocity_) < restThreshold_;
}

void SpringAnimation::update(float deltaTime) {
    if (isAtRest()) return;

    // Spring physics: F = -k * x - c * v
    float displacement = value_ - target_;
    float springForce = -stiffness_ * displacement;
    float dampingForce = -damping_ * velocity_;

    float acceleration = (springForce + dampingForce) / mass_;

    velocity_ += acceleration * deltaTime;
    value_ += velocity_ * deltaTime;
}

// ============================================================================
// AnimationManager Implementation
// ============================================================================

AnimationManager::AnimationManager()
    : lastUpdate_(std::chrono::steady_clock::now()) {
}

AnimationManager& AnimationManager::getInstance() {
    static AnimationManager instance;
    return instance;
}

void AnimationManager::cancelAnimation(AnimationID id) {
    activeAnimations_.erase(
        std::remove_if(activeAnimations_.begin(), activeAnimations_.end(),
                      [id](const AnimationState& state) {
                          return state.id == id;
                      }),
        activeAnimations_.end()
    );
}

void AnimationManager::cancelAllAnimations() {
    activeAnimations_.clear();
}

void AnimationManager::update(float deltaTime) {
    if (paused_) return;

    // Check if reduced motion is enabled
    auto& prefs = Preferences::getInstance();
    if (prefs.accessibility().reducedMotion) {
        deltaTime = 0.0f;  // Skip animations
    }

    // Apply global speed
    deltaTime *= globalSpeed_;

    // Update FPS tracking
    auto now = std::chrono::steady_clock::now();
    float frameDuration = std::chrono::duration<float>(now - lastUpdate_).count();
    lastUpdate_ = now;
    if (frameDuration > 0.0f) {
        avgFPS_ = 0.9f * avgFPS_ + 0.1f * (1.0f / frameDuration);
    }

    // Update all animations
    for (auto& anim : activeAnimations_) {
        anim.updateFunc(deltaTime);
    }

    // Remove completed animations
    activeAnimations_.erase(
        std::remove_if(activeAnimations_.begin(), activeAnimations_.end(),
                      [](const AnimationState& state) {
                          return !state.isActiveFunc();
                      }),
        activeAnimations_.end()
    );
}

// ============================================================================
// Transition Implementation
// ============================================================================

Transition::Transition(Type type, float duration)
    : type_(type), duration_(duration), easing_(Animation::Easing::EaseOut) {
}

void Transition::start() {
    active_ = true;
    reversed_ = false;
    progress_ = 0.0f;
}

void Transition::reverse() {
    reversed_ = !reversed_;
}

void Transition::reset() {
    active_ = false;
    progress_ = 0.0f;
}

void Transition::update(float deltaTime) {
    if (!active_) return;

    float direction = reversed_ ? -1.0f : 1.0f;
    progress_ += (deltaTime / duration_) * direction;
    progress_ = std::clamp(progress_, 0.0f, 1.0f);

    if ((progress_ >= 1.0f && !reversed_) || (progress_ <= 0.0f && reversed_)) {
        active_ = false;
        if (completeCallback_) {
            completeCallback_();
        }
    }
}

Transition::Transform Transition::getTransform() const {
    Transform transform;

    auto easingFunc = Easing::getFunction(easing_);
    float t = easingFunc(progress_);

    switch (type_) {
        case Type::Fade:
            transform.opacity = t;
            break;

        case Type::Slide:
            transform.translateY = (1.0f - t) * 50.0f;  // Slide in from bottom
            transform.opacity = t;
            break;

        case Type::Scale:
            transform.scaleX = 0.8f + 0.2f * t;
            transform.scaleY = 0.8f + 0.2f * t;
            transform.opacity = t;
            break;

        case Type::Rotate:
            transform.rotation = (1.0f - t) * 90.0f;  // Degrees
            transform.opacity = t;
            break;

        case Type::Blur:
            transform.blur = (1.0f - t) * 10.0f;
            transform.opacity = t;
            break;

        case Type::Dissolve:
            transform.opacity = t;
            break;
    }

    return transform;
}

// ============================================================================
// GestureRecognizer Implementation
// ============================================================================

GestureRecognizer::GestureRecognizer()
    : lastTapTime_(std::chrono::steady_clock::now()) {
}

void GestureRecognizer::onMouseDown(float x, float y) {
    Touch touch;
    touch.id = 0;
    touch.x = touch.startX = x;
    touch.y = touch.startY = y;
    touch.startTime = std::chrono::steady_clock::now();

    touches_.push_back(touch);

    // Detect tap/double tap
    auto now = std::chrono::steady_clock::now();
    float timeSinceLastTap =
        std::chrono::duration<float>(now - lastTapTime_).count();

    if (timeSinceLastTap < doubleTapDelay_) {
        if (gestureCallback_) {
            gestureCallback_(GestureType::DoubleTap, x, y);
        }
    }

    lastTapTime_ = now;
}

void GestureRecognizer::onMouseMove(float x, float y) {
    if (!touches_.empty()) {
        touches_[0].x = x;
        touches_[0].y = y;

        // Detect pan
        float dx = x - touches_[0].startX;
        float dy = y - touches_[0].startY;
        float distance = std::sqrt(dx * dx + dy * dy);

        if (distance > 5.0f && gestureCallback_) {
            gestureCallback_(GestureType::Pan, dx, dy);
        }
    }
}

void GestureRecognizer::onMouseUp(float x, float y) {
    if (!touches_.empty()) {
        auto now = std::chrono::steady_clock::now();
        float pressDuration =
            std::chrono::duration<float>(now - touches_[0].startTime).count();

        float dx = x - touches_[0].startX;
        float dy = y - touches_[0].startY;
        float distance = std::sqrt(dx * dx + dy * dy);

        // Detect swipe
        if (distance > swipeThreshold_ && gestureCallback_) {
            gestureCallback_(GestureType::Swipe, dx, dy);
        }
        // Detect tap
        else if (distance < 5.0f && pressDuration < 0.3f && gestureCallback_) {
            gestureCallback_(GestureType::Tap, x, y);
        }
        // Detect long press
        else if (pressDuration >= longPressDelay_ && gestureCallback_) {
            gestureCallback_(GestureType::LongPress, x, y);
        }

        touches_.clear();
    }
}

void GestureRecognizer::onWheel(float deltaX, float deltaY) {
    // Pinch/zoom gesture via wheel
    if (gestureCallback_) {
        gestureCallback_(GestureType::Pinch, deltaX, deltaY);
    }
}

void GestureRecognizer::addTouch(int id, float x, float y) {
    Touch touch;
    touch.id = id;
    touch.x = touch.startX = x;
    touch.y = touch.startY = y;
    touch.startTime = std::chrono::steady_clock::now();
    touches_.push_back(touch);
}

void GestureRecognizer::updateTouch(int id, float x, float y) {
    for (auto& touch : touches_) {
        if (touch.id == id) {
            touch.x = x;
            touch.y = y;
            break;
        }
    }

    // Detect multi-touch gestures
    if (touches_.size() >= 2) {
        // Pinch, rotate, etc.
    }
}

void GestureRecognizer::removeTouch(int id) {
    touches_.erase(
        std::remove_if(touches_.begin(), touches_.end(),
                      [id](const Touch& t) { return t.id == id; }),
        touches_.end()
    );
}

bool GestureRecognizer::detectTap() {
    return false;  // Implemented in onMouseUp
}

bool GestureRecognizer::detectDoubleTap() {
    return false;  // Implemented in onMouseDown
}

bool GestureRecognizer::detectLongPress() {
    return false;  // Implemented in onMouseUp
}

bool GestureRecognizer::detectPan() {
    return false;  // Implemented in onMouseMove
}

bool GestureRecognizer::detectPinch() {
    if (touches_.size() < 2) return false;

    // Calculate distance between two fingers
    float dx = touches_[1].x - touches_[0].x;
    float dy = touches_[1].y - touches_[0].y;
    float currentDistance = std::sqrt(dx * dx + dy * dy);

    float startDx = touches_[1].startX - touches_[0].startX;
    float startDy = touches_[1].startY - touches_[0].startY;
    float startDistance = std::sqrt(startDx * startDx + startDy * startDy);

    float scale = currentDistance / startDistance;

    if (std::abs(scale - 1.0f) > 0.1f && gestureCallback_) {
        gestureCallback_(GestureType::Pinch, scale, 0.0f);
        return true;
    }

    return false;
}

bool GestureRecognizer::detectRotate() {
    if (touches_.size() < 2) return false;

    // Calculate angle between two fingers
    float currentAngle = std::atan2(touches_[1].y - touches_[0].y,
                                   touches_[1].x - touches_[0].x);
    float startAngle = std::atan2(touches_[1].startY - touches_[0].startY,
                                 touches_[1].startX - touches_[0].startX);

    float rotation = currentAngle - startAngle;

    if (std::abs(rotation) > 0.1f && gestureCallback_) {
        gestureCallback_(GestureType::Rotate, rotation * 180.0f / M_PI, 0.0f);
        return true;
    }

    return false;
}

bool GestureRecognizer::detectSwipe() {
    return false;  // Implemented in onMouseUp
}

} // namespace UI
} // namespace MolinAntro
