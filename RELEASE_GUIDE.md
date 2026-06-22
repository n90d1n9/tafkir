# Tafkir Release Guide

This document describes how to create and publish a new release of Tafkir.

## Prerequisites

- JDK 25+ installed
- Access to the `bhangun/tafkir` GitHub repository with write permissions
- Maven Central account (for publishing to Maven Central)
- GPG key configured for signing artifacts

## Version Numbering

Tafkir follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 0.3.0)
- **SNAPSHOT** versions for development (e.g., 0.3.0-SNAPSHOT)

### When to increment:

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

## Release Checklist

### 1. Prepare the Release

- [ ] Ensure all tests pass: `./gradlew test`
- [ ] Update version in `build.gradle.kts` from SNAPSHOT to release version
- [ ] Update `QUICKSTART.md` with any new features or changes
- [ ] Update `README.md` if needed
- [ ] Review and merge all pending PRs
- [ ] Create a release branch (optional for major releases): `git checkout -b release/v0.3.0`

### 2. Test the Build

```bash
# Clean build
./gradlew clean build

# Run all tests
./gradlew test

# Build Javadoc
./gradlew javadocJar

# Build sources jar
./gradlew sourcesJar
```

### 3. Create Git Tag

```bash
# Commit version changes
git add .
git commit -m "Release v0.3.0"

# Create annotated tag
git tag -a v0.3.0 -m "Tafkir v0.3.0"

# Push tag to GitHub
git push origin v0.3.0
```

### 4. Trigger GitHub Release Workflow

The `.github/workflows/release.yml` workflow will automatically:

1. Validate the version
2. Build and test the project
3. Generate Javadoc and sources jars
4. Build native CLI executables for Linux, macOS, and Windows
5. Build JVM package
6. Create GitHub Release with all assets
7. Publish to Maven Central (if configured)

### Manual Release (Workflow Dispatch)

If you need to trigger a release manually:

1. Go to **Actions** → **Release Tafkir**
2. Click **Run workflow**
3. Enter the version number (e.g., `0.3.0`)
4. Click **Run workflow**

### 5. Verify Release

After the workflow completes:

- [ ] Check [GitHub Releases](https://github.com/bhangun/tafkir/releases) for the new release
- [ ] Verify all assets are uploaded:
  - Native binaries (Linux, macOS, Windows)
  - JVM package
  - Javadoc jars
- [ ] Check [Maven Central](https://central.sonatype.com/) for published artifacts (may take 30 minutes to appear)
- [ ] Test installation via JBang: `jbang tafkir@bhangun/tafkir`

### 6. Post-Release Tasks

- [ ] Update version back to SNAPSHOT: `0.4.0-SNAPSHOT`
- [ ] Create release announcement (blog post, social media, etc.)
- [ ] Update documentation website if applicable
- [ ] Notify community (Discord, mailing list, etc.)

## Publishing to Maven Central

### One-Time Setup

1. **Create Sonatype Account**: Register at [OSSRH](https://issues.sonatype.org/)
2. **Create JIRA Ticket**: Request namespace `tech.kayys`
3. **Generate GPG Key**:
   ```bash
   gpg --gen-key
   gpg --armor --export your-email@example.com
   ```
4. **Configure GitHub Secrets**:
   - `MAVEN_CENTRAL_USERNAME`: Your Sonatype username
   - `MAVEN_CENTRAL_PASSWORD`: Your Sonatype password/token
   - `SIGNING_KEY`: Your GPG private key (armored)
   - `SIGNING_PASSWORD`: Your GPG passphrase

### Manual Publishing

```bash
export ORG_GRADLE_PROJECT_mavenCentralUsername=your-username
export ORG_GRADLE_PROJECT_mavenCentralPassword=your-password
export ORG_GRADLE_PROJECT_signingInMemoryKey=$(cat ~/.gnupg/private.key)
export ORG_GRADLE_PROJECT_signingInMemoryKeyPassword=your-passphrase

./gradlew publishToMavenCentral
```

## Troubleshooting

### Build Fails

Check the GitHub Actions logs for specific errors. Common issues:

- **Java version mismatch**: Ensure JDK 25 is used
- **Missing dependencies**: Run `./gradlew dependencies` to diagnose
- **Test failures**: Fix failing tests before release

### Maven Central Publishing Fails

- **Signature validation failed**: Verify GPG key is correctly configured
- **Invalid POM**: Check that all required metadata is present
- **Repository closed**: Wait for Sonatype review (usually 2 hours)

### Native Build Fails

- **Linux**: Install `build-essential`, `libssl-dev`, `zlib1g-dev`
- **macOS**: Install Xcode command line tools and OpenSSL via Homebrew
- **Windows**: Ensure Visual Studio Build Tools are installed

## Emergency Hotfix

For critical bug fixes:

1. Create hotfix branch from release tag: `git checkout -b hotfix/v0.3.1 v0.3.0`
2. Apply fix and update version to `0.3.1`
3. Follow normal release process
4. Merge changes back to main branch

## Contact

For questions about the release process, open an issue or contact the maintainers.
