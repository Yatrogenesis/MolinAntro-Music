#pragma once

#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <mutex>
#include <queue>
#include <chrono>
#include <optional>

namespace MolinAntro {
namespace Cloud {

/**
 * @brief User profile for collaboration
 */
struct User {
    std::string id;
    std::string username;
    std::string displayName;
    std::string email;
    std::string avatarUrl;
    bool isOnline = false;

    enum class Role {
        Owner,      // Full control
        Admin,      // Can manage collaborators
        Editor,     // Can edit project
        Viewer      // Read-only access
    } role = Role::Viewer;

    struct Presence {
        double cursorBeat = 0.0;
        int selectedTrack = -1;
        std::string currentAction;
        uint32_t color = 0xFF5577DD;
    } presence;
};

/**
 * @brief Project change/operation for sync
 */
struct ProjectChange {
    enum class Type {
        // Track operations
        AddTrack,
        RemoveTrack,
        RenameTrack,
        MoveTrack,

        // Clip operations
        AddClip,
        RemoveClip,
        MoveClip,
        ResizeClip,
        EditClipContent,

        // MIDI operations
        AddNote,
        RemoveNote,
        MoveNote,
        ResizeNote,

        // Automation
        AddAutomation,
        EditAutomation,
        RemoveAutomation,

        // Mixer
        SetVolume,
        SetPan,
        SetMute,
        SetSolo,
        AddPlugin,
        RemovePlugin,
        EditPlugin,

        // Global
        SetTempo,
        SetTimeSignature,
        SetLoop,

        // Markers
        AddMarker,
        EditMarker,
        RemoveMarker,

        // File
        AddAudioFile,
        RemoveAudioFile,
        ReplaceAudioFile
    };

    std::string id;                 // Unique change ID
    Type type;
    std::string userId;             // Who made the change
    std::string targetId;           // ID of affected element
    std::map<std::string, std::string> data;  // Change data
    uint64_t timestamp;             // Unix timestamp (ms)
    int sequenceNumber = 0;         // For ordering

    // Conflict resolution
    std::vector<std::string> dependencies;
    bool isResolved = false;
};

/**
 * @brief Comment/annotation on project
 */
struct Comment {
    std::string id;
    std::string userId;
    std::string text;
    uint64_t timestamp;
    double beatPosition;            // Position in timeline
    int trackIndex = -1;            // -1 = project-level
    bool resolved = false;

    // Reply chain
    std::vector<std::string> replyIds;
    std::string parentId;

    // Reactions
    std::map<std::string, std::vector<std::string>> reactions;  // emoji -> userIds
};

/**
 * @brief Project version for history
 */
struct ProjectVersion {
    std::string id;
    std::string name;
    std::string description;
    std::string userId;
    uint64_t timestamp;

    // Snapshot reference
    std::string snapshotUrl;
    size_t snapshotSize = 0;

    // Changes since previous version
    std::vector<std::string> changeIds;

    // Tags
    std::vector<std::string> tags;
};

/**
 * @brief Shared asset (samples, presets, etc.)
 */
struct SharedAsset {
    std::string id;
    std::string name;

    enum class Type {
        AudioSample,
        MIDIPattern,
        Preset,
        Plugin,
        Project,
        Template
    } type = Type::AudioSample;

    std::string uploaderId;
    std::string url;
    size_t sizeBytes = 0;
    std::string format;             // wav, mp3, mid, etc.
    std::string description;
    std::vector<std::string> tags;

    // Usage tracking
    int downloads = 0;
    int usageCount = 0;

    // Audio metadata
    double duration = 0.0;
    double bpm = 0.0;
    std::string key;

    // Licensing
    std::string license;
    bool isRoyaltyFree = true;
};

/**
 * @brief Real-time collaboration session
 */
class CollaborationSession {
public:
    CollaborationSession();
    ~CollaborationSession();

    // Session management
    bool connect(const std::string& projectId, const std::string& authToken);
    void disconnect();
    bool isConnected() const { return connected_; }

    // User management
    void addCollaborator(const std::string& userId, User::Role role);
    void removeCollaborator(const std::string& userId);
    void updateRole(const std::string& userId, User::Role role);
    std::vector<User> getCollaborators() const { return collaborators_; }
    User* getCurrentUser();

    // Presence
    void updatePresence(double cursorBeat, int selectedTrack, const std::string& action);
    std::map<std::string, User::Presence> getPresences() const;

    // Changes
    void submitChange(const ProjectChange& change);
    void applyRemoteChange(const ProjectChange& change);
    std::vector<ProjectChange> getPendingChanges() const;

    // Conflict resolution
    struct ConflictResolution {
        std::string changeId1;
        std::string changeId2;
        enum class Action { KeepLocal, KeepRemote, Merge, Manual } action;
        std::optional<ProjectChange> mergedChange;
    };
    std::vector<ConflictResolution> detectConflicts();
    void resolveConflict(const ConflictResolution& resolution);

    // Comments
    void addComment(const Comment& comment);
    void editComment(const std::string& commentId, const std::string& newText);
    void deleteComment(const std::string& commentId);
    void resolveComment(const std::string& commentId);
    void addReaction(const std::string& commentId, const std::string& emoji);
    std::vector<Comment> getComments() const { return comments_; }

    // Callbacks
    using ChangeCallback = std::function<void(const ProjectChange&)>;
    using PresenceCallback = std::function<void(const std::string& userId, const User::Presence&)>;
    using CommentCallback = std::function<void(const Comment&)>;
    using UserCallback = std::function<void(const User&, bool joined)>;

    void setChangeCallback(ChangeCallback cb) { onChange_ = cb; }
    void setPresenceCallback(PresenceCallback cb) { onPresence_ = cb; }
    void setCommentCallback(CommentCallback cb) { onComment_ = cb; }
    void setUserCallback(UserCallback cb) { onUser_ = cb; }

    // Sync
    void sync();
    bool isSyncing() const { return syncing_; }
    int getPendingChangesCount() const { return static_cast<int>(pendingChanges_.size()); }

private:
    void processIncomingChanges();
    void sendOutgoingChanges();
    bool checkConflict(const ProjectChange& local, const ProjectChange& remote);
    ProjectChange mergeChanges(const ProjectChange& local, const ProjectChange& remote);

    bool connected_ = false;
    bool syncing_ = false;

    std::string projectId_;
    std::string authToken_;
    std::string userId_;

    std::vector<User> collaborators_;
    std::vector<Comment> comments_;

    std::queue<ProjectChange> pendingChanges_;
    std::queue<ProjectChange> incomingChanges_;
    std::vector<ProjectChange> appliedChanges_;

    int localSequence_ = 0;
    int remoteSequence_ = 0;

    ChangeCallback onChange_;
    PresenceCallback onPresence_;
    CommentCallback onComment_;
    UserCallback onUser_;

    mutable std::mutex mutex_;
};

/**
 * @brief Cloud storage manager
 */
class CloudStorage {
public:
    CloudStorage();
    ~CloudStorage();

    // Authentication
    bool login(const std::string& email, const std::string& password);
    bool loginWithToken(const std::string& token);
    void logout();
    bool isLoggedIn() const { return !authToken_.empty(); }
    std::string getAuthToken() const { return authToken_; }

    // Projects
    struct ProjectInfo {
        std::string id;
        std::string name;
        std::string description;
        std::string ownerId;
        uint64_t createdAt;
        uint64_t modifiedAt;
        size_t sizeBytes;
        bool isShared;
        std::vector<std::string> collaboratorIds;
        std::string thumbnailUrl;
        std::vector<std::string> tags;
    };

    std::vector<ProjectInfo> getProjects();
    ProjectInfo createProject(const std::string& name, const std::string& description = "");
    bool deleteProject(const std::string& projectId);
    bool renameProject(const std::string& projectId, const std::string& newName);

    // Upload/Download
    bool uploadProject(const std::string& localPath, const std::string& projectId);
    bool downloadProject(const std::string& projectId, const std::string& localPath);
    float getUploadProgress() const { return uploadProgress_; }
    float getDownloadProgress() const { return downloadProgress_; }
    void cancelTransfer();

    // Assets
    std::string uploadAsset(const SharedAsset& asset, const std::string& localPath);
    bool downloadAsset(const std::string& assetId, const std::string& localPath);
    std::vector<SharedAsset> searchAssets(const std::string& query,
                                           SharedAsset::Type type = SharedAsset::Type::AudioSample,
                                           int limit = 50);

    // Versions
    std::vector<ProjectVersion> getVersions(const std::string& projectId);
    bool createVersion(const std::string& projectId, const std::string& name,
                       const std::string& description);
    bool restoreVersion(const std::string& projectId, const std::string& versionId);

    // Sharing
    std::string createShareLink(const std::string& projectId, bool allowEdit = false,
                                 int expirationDays = 7);
    bool revokeShareLink(const std::string& linkId);
    ProjectInfo getProjectFromShareLink(const std::string& shareToken);

    // Callbacks
    using ProgressCallback = std::function<void(float progress)>;
    void setUploadProgressCallback(ProgressCallback cb) { onUploadProgress_ = cb; }
    void setDownloadProgressCallback(ProgressCallback cb) { onDownloadProgress_ = cb; }

    // Offline support
    void enableOfflineMode(bool enable);
    bool isOfflineMode() const { return offlineMode_; }
    void syncOfflineChanges();
    int getOfflineChangesCount() const;

private:
    std::string authToken_;
    std::string userId_;

    float uploadProgress_ = 0.0f;
    float downloadProgress_ = 0.0f;
    bool transferCancelled_ = false;

    bool offlineMode_ = false;
    std::vector<ProjectChange> offlineChanges_;

    ProgressCallback onUploadProgress_;
    ProgressCallback onDownloadProgress_;

    std::mutex mutex_;
};

/**
 * @brief Version control system for projects
 */
class ProjectVersionControl {
public:
    ProjectVersionControl();
    ~ProjectVersionControl();

    // Initialize for project
    void init(const std::string& projectPath);

    // Commits
    struct Commit {
        std::string id;
        std::string message;
        std::string userId;
        uint64_t timestamp;
        std::string parentId;
        std::vector<std::string> changedFiles;
    };

    std::string commit(const std::string& message);
    std::vector<Commit> getHistory(int limit = 50);
    bool checkout(const std::string& commitId);

    // Branches
    std::string createBranch(const std::string& name);
    bool switchBranch(const std::string& name);
    bool mergeBranch(const std::string& sourceBranch);
    std::vector<std::string> getBranches() const;
    std::string getCurrentBranch() const { return currentBranch_; }

    // Diff
    struct FileDiff {
        std::string path;
        enum class Status { Added, Modified, Deleted } status;
        std::string oldContent;
        std::string newContent;
    };
    std::vector<FileDiff> diff(const std::string& commitId1, const std::string& commitId2);
    std::vector<FileDiff> diffWorking();

    // Stash
    void stash();
    void stashPop();
    bool hasStash() const;

    // Tags
    void addTag(const std::string& commitId, const std::string& tag);
    void removeTag(const std::string& tag);
    std::string getCommitByTag(const std::string& tag);

private:
    std::string projectPath_;
    std::string currentBranch_;
    std::vector<Commit> commits_;
    std::map<std::string, std::string> branches_;  // name -> commitId
    std::map<std::string, std::string> tags_;      // tag -> commitId
    std::vector<FileDiff> stashedChanges_;

    std::string computeHash(const std::string& content);
    void saveState();
    void loadState();
};

/**
 * @brief Real-time chat for collaboration
 */
class CollaborationChat {
public:
    struct Message {
        std::string id;
        std::string userId;
        std::string text;
        uint64_t timestamp;

        enum class Type {
            Text,
            SystemJoin,
            SystemLeave,
            SystemChange,
            FileShare,
            AudioMessage
        } type = Type::Text;

        // For file/audio messages
        std::string attachmentUrl;
        std::string attachmentName;
    };

    CollaborationChat();
    ~CollaborationChat();

    // Connection
    void connect(const std::string& sessionId);
    void disconnect();

    // Messages
    void sendMessage(const std::string& text);
    void sendFileShare(const std::string& fileUrl, const std::string& fileName);
    void sendAudioMessage(const Core::AudioBuffer& audio, int sampleRate);
    std::vector<Message> getMessages(int limit = 100) const;

    // Typing indicator
    void setTyping(bool typing);
    std::vector<std::string> getTypingUsers() const;

    // Callbacks
    using MessageCallback = std::function<void(const Message&)>;
    using TypingCallback = std::function<void(const std::string& userId, bool typing)>;
    void setMessageCallback(MessageCallback cb) { onMessage_ = cb; }
    void setTypingCallback(TypingCallback cb) { onTyping_ = cb; }

private:
    std::string sessionId_;
    std::vector<Message> messages_;
    std::set<std::string> typingUsers_;

    MessageCallback onMessage_;
    TypingCallback onTyping_;

    std::mutex mutex_;
};

/**
 * @brief Community features for sharing
 */
class Community {
public:
    struct Project {
        std::string id;
        std::string name;
        std::string description;
        std::string userId;
        std::string username;
        std::string audioPreviewUrl;
        std::string thumbnailUrl;
        uint64_t publishedAt;

        int plays = 0;
        int likes = 0;
        int comments = 0;
        int forks = 0;

        std::vector<std::string> tags;
        std::string genre;
        double bpm = 0.0;
        std::string key;
    };

    struct UserProfile {
        std::string id;
        std::string username;
        std::string displayName;
        std::string bio;
        std::string avatarUrl;
        std::string location;
        std::string website;

        int followers = 0;
        int following = 0;
        int projectCount = 0;
        int totalPlays = 0;

        std::vector<std::string> genres;
        bool isVerified = false;
    };

    Community();
    ~Community();

    // Discovery
    std::vector<Project> getFeatured(int limit = 20);
    std::vector<Project> getTrending(int limit = 20);
    std::vector<Project> getRecent(int limit = 20);
    std::vector<Project> searchProjects(const std::string& query, int limit = 50);
    std::vector<Project> getByGenre(const std::string& genre, int limit = 50);
    std::vector<Project> getByUser(const std::string& userId, int limit = 50);

    // Project interaction
    void play(const std::string& projectId);
    void like(const std::string& projectId);
    void unlike(const std::string& projectId);
    std::string fork(const std::string& projectId);  // Returns new project ID
    void addComment(const std::string& projectId, const std::string& text);

    // Publishing
    void publishProject(const std::string& localProjectId, const Project& metadata);
    void unpublishProject(const std::string& projectId);
    void updatePublishedProject(const std::string& projectId, const Project& metadata);

    // User profiles
    UserProfile getProfile(const std::string& userId);
    void updateProfile(const UserProfile& profile);
    void follow(const std::string& userId);
    void unfollow(const std::string& userId);
    std::vector<UserProfile> getFollowers(const std::string& userId, int limit = 50);
    std::vector<UserProfile> getFollowing(const std::string& userId, int limit = 50);

    // Feed
    std::vector<Project> getFeed(int limit = 50);  // Projects from followed users

    // Notifications
    struct Notification {
        std::string id;
        enum class Type {
            NewFollower,
            ProjectLiked,
            ProjectCommented,
            ProjectForked,
            NewProjectFromFollowing,
            CollaborationInvite,
            MentionInComment
        } type;
        std::string fromUserId;
        std::string projectId;
        std::string text;
        uint64_t timestamp;
        bool read = false;
    };
    std::vector<Notification> getNotifications(int limit = 50);
    void markNotificationRead(const std::string& notificationId);
    void markAllNotificationsRead();
    int getUnreadCount();

private:
    std::string authToken_;
    std::vector<Notification> notifications_;
    std::mutex mutex_;
};

/**
 * @brief Project export/render to cloud
 */
class CloudExporter {
public:
    struct ExportSettings {
        enum class Format {
            WAV_16,
            WAV_24,
            WAV_32F,
            MP3_128,
            MP3_192,
            MP3_320,
            FLAC,
            OGG,
            AAC,
            AIFF
        } format = Format::WAV_24;

        int sampleRate = 48000;
        bool normalize = true;
        float normalizeLevel = -1.0f;  // dB (peak)
        bool dither = true;

        // Stems export
        bool exportStems = false;
        std::vector<std::string> stemGroups;  // Track groups for stems

        // Metadata
        std::string title;
        std::string artist;
        std::string album;
        std::string year;
        std::string genre;
        std::string comment;
        std::string copyright;
    };

    struct ExportResult {
        bool success = false;
        std::string url;
        std::string errorMessage;
        size_t fileSize = 0;
        double duration = 0.0;
    };

    CloudExporter();
    ~CloudExporter();

    // Export to cloud
    ExportResult exportToCloud(const Core::AudioBuffer& audio, int sampleRate,
                                const ExportSettings& settings);

    // Progress
    float getProgress() const { return progress_; }
    void cancel();

    // Callbacks
    using ProgressCallback = std::function<void(float progress, const std::string& stage)>;
    void setProgressCallback(ProgressCallback cb) { onProgress_ = cb; }

private:
    float progress_ = 0.0f;
    bool cancelled_ = false;
    ProgressCallback onProgress_;

    std::mutex mutex_;
};

} // namespace Cloud
} // namespace MolinAntro
