# Local privacy and encryption design

## Windows laptop development

Use a dedicated local data directory outside the checkout on a Windows volume
protected by Device Encryption or BitLocker where your Windows edition supports
it. Check encryption status with Windows Settings or `manage-bde -status`; do not
assume that enabling Python file modes sets Windows ACLs or encrypts data. Restrict
the directory's ACL to the account running the application and authorized local
administrators. Do not use a shared/synced folder for templates or recordings.

Volume keys should be managed by Windows/TPM, with recovery material held securely
outside the repository and away from the data volume. No application key, password,
recovery code or TPM credential belongs in CLI arguments, source, logs or Git.
This sprint specifies OS-managed encryption; it does not implement or automatically
verify application-level encryption. A locked Windows session is not protection
against a privileged process or an unlocked disk mounted by the same account.

For deployment, document the threat model, device encryption/boot policy, account
and ACL provisioning, recovery-key custody, encrypted backups, rotation/recovery,
and device disposal. If per-file application encryption becomes necessary, use an
established AEAD construction, separately managed keys protected by the OS, explicit
format versions and crash-safe migration; do not invent custom cryptography.

## Collection and deletion

Unfamiliar face crop collection is **off by default**. `monitor --save-crops` is an
explicit opt-in, not consent management. Embeddings and identity/event metadata are
still sensitive even when image collection is disabled. Do not capture audio/video.

`profile-delete --name NAME --yes` removes that gallery entry and by default purges
its known events and all events in linked episodes. This includes classification
transitions in the same episode. `--keep-events` explicitly retains metadata.
Unassociated unfamiliar events cannot be attributed to a deleted person reliably.
Stop any enrollment capture for that profile before deletion; a deliberate later
enrollment can otherwise recreate it. Cooperating monitoring frames hold the gallery
lock and reload changed galleries before inference, so deletion does not leave a
permanently cached identity. The command may wait for an in-flight frame to finish.

Deletion is idempotent. Gallery removal precedes event purging. If purging fails,
rerun the command: removed templates are not restored. Atomic NPZ replacement and
SQLite deletion do not guarantee physical erasure from SSDs, journals, backups,
page caches or snapshots. OS encryption and backup/disposal policy are necessary.

`maintain --yes` runs retention and orphan recovery. Add `--drop-crops` to remove
existing generated crops without deleting event rows; stop crop-producing monitors
first if all crops must stay absent. Event listing hides names and image paths by
default; `events --show-identities` explicitly displays local sensitive fields.

## Budgets and retention

Monitoring defaults to 7-day/10,000-event periodic retention and a 512 MiB storage
admission budget. The budget counts local gallery, database and crop files, with a
64 KiB reserve before an event write. It refuses new events when over budget and
does not advance cooldown on failure. It is not a filesystem quota: concurrent
noncooperating writers, SQLite journal overhead and external backups are outside
this estimate. Use a filesystem quota/free-space monitoring for a hard bound.
Closed/unreferenced episode metadata ages out with event retention. No data is
silently uploaded. Synthetic test fixtures are generated under temporary directories.
