; Script generated for Inno Setup 6.x
; MalevolentSlice Studio - Windows Setup Installer Script
; See: https://jrsoftware.org/isinfo.php

#define MyAppName "MalevolentSlice Studio"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "Fidriyani"
#define MyAppURL "https://github.com/drrri-py/malevolentslice"
#define MyAppExeName "MalevolentSliceStudio.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{94B817E2-60D9-45B1-8C7E-2F38C15488EA}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; Default output installer name and directory
OutputDir=dist\installer
OutputBaseFilename=MalevolentSliceStudio_Setup_v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Distribute all files from PyInstaller onedir output
Source: "dist\MalevolentSliceStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
