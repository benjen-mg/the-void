; Inno Setup script: wraps the PyInstaller folder (dist\the Void) into
; "the-Void-<version>-windows-setup.exe". The release workflow runs
;
;     ISCC.exe /DAppVersion=1.0.0 packaging\the_void.iss
;
; from the repository root, after PyInstaller and the self-test.
;
; Installs for the current user by default, so no administrator rights are
; needed; the installer offers "install for all users" as well.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; Fixed for good: Windows uses it to recognise later versions as upgrades
; of this one. Never change it.
AppId={{146A5CBD-90AE-4A68-84C1-75EF6E5C5417}
AppName=the Void
AppVersion={#AppVersion}
AppVerName=the Void {#AppVersion}
AppPublisher=Benyamin M. Garmejani
AppPublisherURL=https://github.com/benjen-mg/the-void
AppSupportURL=https://github.com/benjen-mg/the-void/issues
AppUpdatesURL=https://github.com/benjen-mg/the-void/releases
AppCopyright=Copyright (c) 2026 Benyamin M. Garmejani
DefaultDirName={autopf}\the Void
DefaultGroupName=the Void
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=the-Void-{#AppVersion}-windows-setup
SetupIconFile=..\thevoid_core\assets\icon.ico
UninstallDisplayIcon={app}\the Void.exe
UninstallDisplayName=the Void {#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
VersionInfoVersion={#AppVersion}
VersionInfoProductName=the Void
VersionInfoDescription=the Void installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[InstallDelete]
; An upgrade replaces the previous version's files rather than mixing them.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\the Void\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\the Void"; Filename: "{app}\the Void.exe"
Name: "{autoprograms}\the Void example structure"; Filename: "{app}\_internal\examples"
Name: "{autodesktop}\the Void"; Filename: "{app}\the Void.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\the Void.exe"; Description: "Start the Void now"; Flags: nowait postinstall skipifsilent
