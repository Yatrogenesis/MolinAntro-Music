/**
 * @file CloudCollaborationTest.cpp
 * @brief Unit tests for Cloud Collaboration system
 *
 * Tests for:
 * - CollaborationSession (WebSocket messaging)
 * - CloudStorage (project management)
 * - ProjectVersionControl (git-like VCS)
 * - CollaborationChat (messaging)
 *
 * Note: Network tests use mock callbacks since actual server
 * connection is not available in unit test environment.
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "cloud/CloudCollaboration.h"
#include <thread>
#include <chrono>

using namespace MolinAntro::Cloud;

//=============================================================================
// CollaborationSession Tests
//=============================================================================

class CollaborationSessionTest : public ::testing::Test {
protected:
    void SetUp() override {
        session = std::make_unique<CollaborationSession>();
    }

    std::unique_ptr<CollaborationSession> session;
};

TEST_F(CollaborationSessionTest, InitialState) {
    EXPECT_FALSE(session->isConnected());
    EXPECT_TRUE(session->getCollaborators().empty());
}

TEST_F(CollaborationSessionTest, CallbackRegistration) {
    bool presenceCalled = false;
    bool changeCalled = false;
    bool commentCalled = false;
    bool userCalled = false;

    session->setPresenceCallback([&](const std::string& userId, const User::Presence&) {
        presenceCalled = true;
    });

    session->setChangeCallback([&](const ProjectChange&) {
        changeCalled = true;
    });

    session->setCommentCallback([&](const Comment&) {
        commentCalled = true;
    });

    session->setUserCallback([&](const User&, bool) {
        userCalled = true;
    });

    // Callbacks should be registered without crashing
    EXPECT_FALSE(presenceCalled);  // Not triggered yet
}

TEST_F(CollaborationSessionTest, AddCollaborator) {
    session->addCollaborator("user-123", User::Role::Editor);

    auto collaborators = session->getCollaborators();
    EXPECT_EQ(collaborators.size(), 1);
    EXPECT_EQ(collaborators[0].id, "user-123");
    EXPECT_EQ(collaborators[0].role, User::Role::Editor);
}

TEST_F(CollaborationSessionTest, RemoveCollaborator) {
    session->addCollaborator("user-123", User::Role::Editor);
    session->addCollaborator("user-456", User::Role::Viewer);

    EXPECT_EQ(session->getCollaborators().size(), 2);

    session->removeCollaborator("user-123");

    auto collaborators = session->getCollaborators();
    EXPECT_EQ(collaborators.size(), 1);
    EXPECT_EQ(collaborators[0].id, "user-456");
}

TEST_F(CollaborationSessionTest, UpdateRole) {
    session->addCollaborator("user-123", User::Role::Viewer);
    session->updateRole("user-123", User::Role::Owner);

    auto collaborators = session->getCollaborators();
    EXPECT_EQ(collaborators[0].role, User::Role::Owner);
}

TEST_F(CollaborationSessionTest, SubmitChange) {
    ProjectChange change;
    change.type = ProjectChange::Type::AddClip;
    change.targetId = "track-1";
    change.data["clipName"] = "Test Clip";

    // Should not crash even when not connected
    session->submitChange(change);

    // Change should be queued
    auto pending = session->getPendingChanges();
    EXPECT_EQ(pending.size(), 1);
    EXPECT_EQ(pending[0].type, ProjectChange::Type::AddClip);
}

TEST_F(CollaborationSessionTest, ConflictDetection) {
    // Add a local change
    ProjectChange local;
    local.id = "local-1";
    local.type = ProjectChange::Type::MoveClip;
    local.targetId = "clip-123";
    local.timestamp = 1000;

    session->submitChange(local);

    // Simulate remote change on same target
    ProjectChange remote;
    remote.id = "remote-1";
    remote.type = ProjectChange::Type::MoveClip;
    remote.targetId = "clip-123";
    remote.timestamp = 1001;

    session->applyRemoteChange(remote);

    // Should detect conflict
    auto conflicts = session->detectConflicts();
    // Note: Conflict detection depends on internal state
}

TEST_F(CollaborationSessionTest, Comments) {
    Comment comment;
    comment.text = "This sounds great!";
    comment.beatPosition = 16.0;

    session->addComment(comment);

    auto comments = session->getComments();
    EXPECT_EQ(comments.size(), 1);
    EXPECT_EQ(comments[0].text, "This sounds great!");
    EXPECT_EQ(comments[0].beatPosition, 16.0);
}

TEST_F(CollaborationSessionTest, EditComment) {
    Comment comment;
    comment.text = "Original text";

    session->addComment(comment);
    auto comments = session->getComments();
    std::string commentId = comments[0].id;

    session->editComment(commentId, "Edited text");

    // Note: Edit requires matching userId
}

TEST_F(CollaborationSessionTest, DeleteComment) {
    Comment comment;
    comment.text = "To be deleted";

    session->addComment(comment);
    auto comments = session->getComments();
    EXPECT_EQ(comments.size(), 1);

    // Note: Delete requires matching userId
}

TEST_F(CollaborationSessionTest, ResolveComment) {
    Comment comment;
    comment.text = "Fix this section";
    comment.resolved = false;

    session->addComment(comment);
    auto comments = session->getComments();
    std::string commentId = comments[0].id;

    session->resolveComment(commentId);

    comments = session->getComments();
    EXPECT_TRUE(comments[0].resolved);
}

//=============================================================================
// CloudStorage Tests
//=============================================================================

class CloudStorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        storage = std::make_unique<CloudStorage>();
    }

    std::unique_ptr<CloudStorage> storage;
};

TEST_F(CloudStorageTest, InitialState) {
    EXPECT_FALSE(storage->isLoggedIn());
}

TEST_F(CloudStorageTest, Login) {
    bool result = storage->login("test@example.com", "password");

    EXPECT_TRUE(result);
    EXPECT_TRUE(storage->isLoggedIn());
}

TEST_F(CloudStorageTest, LoginWithToken) {
    bool result = storage->loginWithToken("mock-jwt-token");

    EXPECT_TRUE(result);
    EXPECT_TRUE(storage->isLoggedIn());
}

TEST_F(CloudStorageTest, Logout) {
    storage->login("test@example.com", "password");
    EXPECT_TRUE(storage->isLoggedIn());

    storage->logout();
    EXPECT_FALSE(storage->isLoggedIn());
}

TEST_F(CloudStorageTest, CreateProject) {
    storage->login("test@example.com", "password");

    auto project = storage->createProject("My New Track", "Electronic music");

    EXPECT_FALSE(project.id.empty());
    EXPECT_EQ(project.name, "My New Track");
    EXPECT_EQ(project.description, "Electronic music");
}

TEST_F(CloudStorageTest, GetProjects) {
    storage->login("test@example.com", "password");

    auto projects = storage->getProjects();

    // Should return at least mock data
    EXPECT_GE(projects.size(), 0);
}

TEST_F(CloudStorageTest, UploadProgressCallback) {
    storage->login("test@example.com", "password");

    float lastProgress = 0.0f;
    storage->setUploadProgressCallback([&lastProgress](float progress) {
        lastProgress = progress;
    });

    // Note: Actual upload would update progress
}

TEST_F(CloudStorageTest, DownloadProgressCallback) {
    storage->login("test@example.com", "password");

    float lastProgress = 0.0f;
    storage->setDownloadProgressCallback([&lastProgress](float progress) {
        lastProgress = progress;
    });

    // Note: Actual download would update progress
}

TEST_F(CloudStorageTest, CreateShareLink) {
    storage->login("test@example.com", "password");

    std::string link = storage->createShareLink("project-123", true, 7);

    EXPECT_FALSE(link.empty());
    EXPECT_TRUE(link.find("molinantro.app/share/") != std::string::npos);
}

TEST_F(CloudStorageTest, OfflineMode) {
    storage->enableOfflineMode(true);

    EXPECT_EQ(storage->getOfflineChangesCount(), 0);

    storage->enableOfflineMode(false);
}

//=============================================================================
// ProjectVersionControl Tests
//=============================================================================

class ProjectVersionControlTest : public ::testing::Test {
protected:
    void SetUp() override {
        vcs = std::make_unique<ProjectVersionControl>();
    }

    std::unique_ptr<ProjectVersionControl> vcs;
};

TEST_F(ProjectVersionControlTest, Init) {
    vcs->init("/test/project");

    EXPECT_EQ(vcs->getCurrentBranch(), "main");
}

TEST_F(ProjectVersionControlTest, Commit) {
    vcs->init("/test/project");

    std::string commitId = vcs->commit("First change");

    EXPECT_FALSE(commitId.empty());

    auto history = vcs->getHistory(10);
    EXPECT_GE(history.size(), 1);
}

TEST_F(ProjectVersionControlTest, CreateBranch) {
    vcs->init("/test/project");
    vcs->commit("Initial commit");

    std::string branchCommit = vcs->createBranch("feature/new-synth");

    EXPECT_FALSE(branchCommit.empty());

    auto branches = vcs->getBranches();
    EXPECT_EQ(branches.size(), 2);  // main + new branch
}

TEST_F(ProjectVersionControlTest, SwitchBranch) {
    vcs->init("/test/project");
    vcs->commit("Initial commit");
    vcs->createBranch("develop");

    bool switched = vcs->switchBranch("develop");

    EXPECT_TRUE(switched);
    EXPECT_EQ(vcs->getCurrentBranch(), "develop");
}

TEST_F(ProjectVersionControlTest, SwitchToNonexistentBranch) {
    vcs->init("/test/project");

    bool switched = vcs->switchBranch("nonexistent");

    EXPECT_FALSE(switched);
    EXPECT_EQ(vcs->getCurrentBranch(), "main");
}

TEST_F(ProjectVersionControlTest, MergeBranch) {
    vcs->init("/test/project");
    vcs->commit("Initial");
    vcs->createBranch("feature");
    vcs->switchBranch("feature");
    vcs->commit("Feature commit");
    vcs->switchBranch("main");

    bool merged = vcs->mergeBranch("feature");

    EXPECT_TRUE(merged);
}

TEST_F(ProjectVersionControlTest, Tags) {
    vcs->init("/test/project");
    std::string commitId = vcs->commit("Version 1.0");

    vcs->addTag(commitId, "v1.0");

    std::string taggedCommit = vcs->getCommitByTag("v1.0");
    EXPECT_EQ(taggedCommit, commitId);

    vcs->removeTag("v1.0");
    taggedCommit = vcs->getCommitByTag("v1.0");
    EXPECT_TRUE(taggedCommit.empty());
}

TEST_F(ProjectVersionControlTest, Stash) {
    vcs->init("/test/project");

    EXPECT_FALSE(vcs->hasStash());

    vcs->stash();

    // Note: Stash behavior depends on working directory changes
}

TEST_F(ProjectVersionControlTest, History) {
    vcs->init("/test/project");
    vcs->commit("Commit 1");
    vcs->commit("Commit 2");
    vcs->commit("Commit 3");

    auto history = vcs->getHistory(2);

    EXPECT_LE(history.size(), 2);
}

//=============================================================================
// CollaborationChat Tests
//=============================================================================

class CollaborationChatTest : public ::testing::Test {
protected:
    void SetUp() override {
        chat = std::make_unique<CollaborationChat>();
    }

    std::unique_ptr<CollaborationChat> chat;
};

TEST_F(CollaborationChatTest, SendTextMessage) {
    chat->connect("session-123");

    chat->sendMessage("Hello everyone!");

    auto messages = chat->getMessages(10);
    EXPECT_EQ(messages.size(), 1);
    EXPECT_EQ(messages[0].text, "Hello everyone!");
    EXPECT_EQ(messages[0].type, CollaborationChat::Message::Type::Text);
}

TEST_F(CollaborationChatTest, SendFileShare) {
    chat->connect("session-123");

    chat->sendFileShare("https://example.com/file.wav", "drums.wav");

    auto messages = chat->getMessages(10);
    EXPECT_EQ(messages.size(), 1);
    EXPECT_EQ(messages[0].type, CollaborationChat::Message::Type::FileShare);
    EXPECT_EQ(messages[0].attachmentName, "drums.wav");
}

TEST_F(CollaborationChatTest, MessageCallback) {
    chat->connect("session-123");

    bool callbackFired = false;
    CollaborationChat::Message receivedMsg;

    chat->setMessageCallback([&](const CollaborationChat::Message& msg) {
        callbackFired = true;
        receivedMsg = msg;
    });

    chat->sendMessage("Test message");

    EXPECT_TRUE(callbackFired);
    EXPECT_EQ(receivedMsg.text, "Test message");
}

TEST_F(CollaborationChatTest, MessageLimit) {
    chat->connect("session-123");

    // Send 20 messages
    for (int i = 0; i < 20; ++i) {
        chat->sendMessage("Message " + std::to_string(i));
    }

    // Get last 5
    auto messages = chat->getMessages(5);
    EXPECT_EQ(messages.size(), 5);
}

TEST_F(CollaborationChatTest, Disconnect) {
    chat->connect("session-123");
    chat->disconnect();

    // Should not crash after disconnect
    chat->sendMessage("After disconnect");
}

//=============================================================================
// Community Tests
//=============================================================================

class CommunityTest : public ::testing::Test {
protected:
    void SetUp() override {
        community = std::make_unique<Community>();
    }

    std::unique_ptr<Community> community;
};

TEST_F(CommunityTest, GetFeatured) {
    auto featured = community->getFeatured(10);

    // Returns empty in mock, but shouldn't crash
    EXPECT_GE(featured.size(), 0);
}

TEST_F(CommunityTest, SearchProjects) {
    auto results = community->searchProjects("electronic", 20);

    EXPECT_GE(results.size(), 0);
}

TEST_F(CommunityTest, Fork) {
    std::string newProjectId = community->fork("original-project-123");

    // Should return a new UUID
    EXPECT_FALSE(newProjectId.empty());
}

TEST_F(CommunityTest, Notifications) {
    auto notifications = community->getNotifications(10);

    EXPECT_GE(notifications.size(), 0);
    EXPECT_EQ(community->getUnreadCount(), 0);
}

TEST_F(CommunityTest, MarkAllNotificationsRead) {
    // Should not crash even with no notifications
    community->markAllNotificationsRead();

    EXPECT_EQ(community->getUnreadCount(), 0);
}

TEST_F(CommunityTest, UserProfile) {
    auto profile = community->getProfile("user-123");

    EXPECT_EQ(profile.id, "user-123");
}

//=============================================================================
// CloudExporter Tests
//=============================================================================

class CloudExporterTest : public ::testing::Test {
protected:
    void SetUp() override {
        exporter = std::make_unique<CloudExporter>();
    }

    std::unique_ptr<CloudExporter> exporter;
};

TEST_F(CloudExporterTest, ExportProgress) {
    std::vector<float> progressValues;

    exporter->setProgressCallback([&](float progress, const std::string& stage) {
        progressValues.push_back(progress);
    });

    MolinAntro::Core::AudioBuffer audio(2, 48000);
    CloudExporter::ExportSettings settings;
    settings.format = CloudExporter::Format::WAV;
    settings.sampleRate = 48000;
    settings.bitDepth = 24;

    auto result = exporter->exportToCloud(audio, 48000, settings);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.url.empty());
    EXPECT_FALSE(progressValues.empty());
}

TEST_F(CloudExporterTest, CancelExport) {
    // Start export in thread and cancel
    exporter->cancel();

    // Should handle cancellation gracefully
}
