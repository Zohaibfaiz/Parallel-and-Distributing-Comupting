<#
 Static IP for the direct Ethernet link (Windows). Run in an *Administrator* PowerShell:

   .\scripts\setup_static_ip_windows.ps1 -Role server -Interface "Ethernet"   # 192.168.1.1
   .\scripts\setup_static_ip_windows.ps1 -Role client -Interface "Ethernet"   # 192.168.1.2
   .\scripts\setup_static_ip_windows.ps1 -Role reset  -Interface "Ethernet"   # back to DHCP

 List your adapter names with:  Get-NetAdapter
 The worker (server) also gets a firewall rule for TCP 5050 and ICMP echo (ping).
#>
param(
  [Parameter(Mandatory = $true)][ValidateSet('server', 'client', 'reset')][string]$Role,
  [string]$Interface = 'Ethernet',
  [int]$Port = 5050
)
$ErrorActionPreference = 'Stop'
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
      [Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Error 'Please run this script as Administrator.'; exit 1
}

Get-NetIPAddress -InterfaceAlias $Interface -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
Remove-NetRoute -InterfaceAlias $Interface -Confirm:$false -ErrorAction SilentlyContinue

if ($Role -eq 'reset') {
  Set-NetIPInterface -InterfaceAlias $Interface -Dhcp Enabled
  Write-Host "Interface '$Interface' is back on DHCP."; exit 0
}

$ip = if ($Role -eq 'server') { '192.168.1.1' } else { '192.168.1.2' }
Set-NetIPInterface -InterfaceAlias $Interface -Dhcp Disabled
New-NetIPAddress -InterfaceAlias $Interface -IPAddress $ip -PrefixLength 24 | Out-Null   # no gateway needed
Set-NetConnectionProfile -InterfaceAlias $Interface -NetworkCategory Private -ErrorAction SilentlyContinue

if ($Role -eq 'server') {
  if (-not (Get-NetFirewallRule -DisplayName 'GPU Offload Worker' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName 'GPU Offload Worker' -Direction Inbound -Protocol TCP -LocalPort $Port `
      -Action Allow -Profile Private | Out-Null
  }
  if (-not (Get-NetFirewallRule -DisplayName 'GPU Offload ICMP' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName 'GPU Offload ICMP' -Direction Inbound -Protocol ICMPv4 -IcmpType 8 `
      -Action Allow -Profile Private | Out-Null
  }
}
Get-NetIPAddress -InterfaceAlias $Interface -AddressFamily IPv4 | Format-Table IPAddress, PrefixLength, InterfaceAlias
Write-Host "Done. This machine is now $ip/24 ($Role). Test from the other PC:  ping $(if ($Role -eq 'server') {'192.168.1.1'} else {'192.168.1.2'})"
